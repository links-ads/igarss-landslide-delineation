from src.misc import nglob
import albumentations as A
from tqdm import tqdm
from src.io_utils import read_raster
from src.augs import RandomBrightnessContrast
import os
import cv2
from glob import glob
import numpy as np
import rasterio as rio
import pandas as pd
#import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
#from einops import rearrange
from torch.utils.data import Dataset, Subset
from torchvision.utils import draw_segmentation_masks
#from src.transforms import transforms
#from src.utils import dataset_split, natural_sort



def load_landslide_df(path):
    """
    Compose a dataframe of all S2 products
    """

    pre_imgs = nglob(os.path.join(path, 'pre', '*.tif'))
    post_imgs = nglob(os.path.join(path, 'post', '*.tif'))
    dem = os.path.join(path, 'dem.tif')
    slope = os.path.join(path, 'slope.tif')
    aspect = os.path.join(path, 'aspect.tif')
    polygons = os.path.join(path, 'polygons.tif')

    pre_df = pd.DataFrame({
        'time': 'pre',
        'img': pre_imgs,
        'dem': dem,
        'slope': slope,
        'aspect': aspect,
        'polygons': polygons,
    })
    post_df = pd.DataFrame({
        'time': 'post',
        'img': post_imgs,
        'dem': dem,
        'slope': slope,
        'aspect': aspect,
        'polygons': polygons,
    })

    return pd.concat([pre_df, post_df])



class PatchDataset(Dataset):
    """
    Dataset of patch couples (pre and post disaster images)

    Args:
        df (pd.DataFrame): dataframe with patches metadata
        augment_common (albumentations.Compose): data transfromation pipeline
        common between pre- and post- images
        augment_individual (albumentations.Compose): data transfromation
        pipeline for each image
        prefetch (bool): if True, prefetch images in memory
        bands (list): list of bands to read from post disaster images
        ignore_cloud_mask_classes (list): list of cloud detector's classes to
        ignore in the loss computation
    """

    transform_targets = {
        "pre"    : "image",
        "post"   : "image",
        "dem"    : "mask",
        "slope"  : "mask",
        "aspect" : "mask",
        "mask"   : "mask",
    }
    # NB: albumentations needs at least 'image' and 'mask' keys.
    # See _preprocess() and _postprocess()

    default_train_transforms_common = A.ReplayCompose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=1),
    #A.RandomResizedCrop(height=256, width=256, scale=(0.5, 1.0), ratio=(0.9, 1.1), p=0.3),
    ])  # geometry-disrupting augmentations -> to be applied to both pre and post with the same parameters

    default_train_transforms_individual = A.Compose([
        A.HistogramMatching(
            reference_images = [],  # these will be filled in the __init__ method with idxs of the reference images
            blend_ratio = (0.3,0.7),
            read_fn = lambda _: None, # this will be initialized in the __init__ method with a lambda function that reads the reference images given their idxs
            p = 0.4),  # to add robustness to seasonal changes
        RandomBrightnessContrast(brightness_limit=(-0.8,1.6), contrast_limit=(0.0,1.0), contrast_factor=1000, p=0.7),
        #A.ChannelDropout(channel_drop_range=(1, 2), fill_value=0, p=0.2),  # to force the model to not rely on a single band
        #A.ChannelShuffle(p=0.2),  # to force the network to learn the spatial relationships between bands instead of the spectral ones
        A.Affine(translate_px = {'x':(0,1),'y':(0,1)}, rotate = (-1,1), mode = cv2.BORDER_REFLECT, p = 0.2),  # to add robustness to inexact co-registration of the pre- and post- images, as done in Dahitra paper
        ])  # geometry-preserving augmentations -> can be applied independently to pre and post with different parameters
    
    default_eval_transforms_common = A.ReplayCompose([
        A.NoOp(p=1),
        ])
    
    default_eval_transforms_individual = A.Compose([
        A.NoOp(p=1),
        ])

    def __init__(
            self,
            df,
            augment_common=None,
            augment_individual=None,
            prefetch=False,
            bands=None,
            cloud_cover_to_mask={0:1, 1:-1, 2:-1, 3:1},
            n_postpost_pairs=0,
            n_postpre_pairs=0,
            n_prepre_pairs=0,
            ohe_mask=False,
            root='',
            verbose=True,
            ):
        self.df = df.drop_duplicates()
        self.prods = [col.split("_")[1] for col in df.columns if col.startswith("fn_")]

        self.augment_common = augment_common or self.default_train_transforms_common
        self.augment_common.add_targets(self.transform_targets)
        self.augment_individual = augment_individual or self.default_train_transforms_individual
        for t in self.augment_individual:
            if isinstance(t, A.HistogramMatching):
                t.reference_images = list(range(len(self.df)))
                t.read_fn = lambda idx: self.read(idx)[np.random.choice(['pre','post'])].astype(np.float32).transpose(1,2,0)

        self.prefetch = prefetch    # if True, prefetch images in memory
        if bands is not None:
            assert bands[-1] == 'cloud_mask', "cloud_mask must be the last band"
        self.bands = bands # S2 bands to keep (if None, keep all)

        #self.ignore_cloud_mask_classes = ignore_cloud_mask_classes
        if isinstance(cloud_cover_to_mask, list):
            assert len(cloud_cover_to_mask) == 4, f"cloud_cover_to_mask list must have length 4. Got length = {len(cloud_cover_to_mask)}"
            self.cloud_cover_to_mask = {i : cloud_cover_to_mask[i] for i in range(4)}
        elif isinstance(cloud_cover_to_mask, dict):
            self.cloud_cover_to_mask = cloud_cover_to_mask
        else:
            raise TypeError(f"cloud_cover_to_mask must be either a list or a dict. Got type = {type(cloud_cover_to_mask)}")
        # KEYS: 0: clear sky, 1: thick cloud, 2: thin cloud, 3: cloud shadow; VALUES: 1: visible, 0: not visible, -1: ignore

        self.ohe_mask = ohe_mask
        self.root = root
        self.verbose = verbose
    
        if self.prefetch:
            # prefetch images
            # (WARNING: memory-intensive, but time-saving in the long run)
            if self.verbose: print('Prefetching images...')
            self._cache = {}   # images cache
            
            for prod in self.prods:

                # initialize patch ids
                self.df[f'{prod}_patch_id'] = ""
                    
                # to avoid patch duplicates, group by patch coordinates for each
                # product and load each patch only once
                groupby_keys = [f'fn_{prod}','h_start','h_end','w_start','w_end']
                
                for i, (_, rows) in enumerate(tqdm(self.df.groupby(by=groupby_keys), desc=prod, disable=not self.verbose)):

                    # assign patch index
                    patch_id = f"{prod}_{i}"    # e.g. "pre_5" or "poly_19"
                    self.df.loc[rows.index,f'{prod}_patch_id'] = patch_id

                    # read patch
                    path = os.path.join(self.root, rows.iloc[0][f'fn_{prod}'])
                    window = rows.iloc[0][['h_start','h_end','w_start','w_end']].to_numpy()
                    if prod in ['pre','post']:
                        raster = read_raster(path, window=window, bands=self.bands)
                        raster, cm = raster[:-1], raster[[-1]]
                        # S2 image
                        self._cache[patch_id] = raster
                        # cloud mask (needs another patch index)
                        patch_cm_id = f"{prod}_cm_{i}"
                        self.df.loc[rows.index,f'{prod}_cm_patch_id'] = patch_cm_id
                        self._cache[patch_cm_id] = cm
                    else:
                        raster = read_raster(path, window=window)
                        self._cache[patch_id] = raster
                    


        
        # generate artificial pairs
        self.original_len = len(self.df)
        self.n_postpost_pairs = n_postpost_pairs
        self.n_postpre_pairs = n_postpre_pairs
        self.n_prepre_pairs = n_prepre_pairs
        self.regenerate_artificial_pairs()
    
    def regenerate_artificial_pairs(self):
        self.postpost_df = self._generate_artificial_pairs(self.n_postpost_pairs, mode='postpost')
        self.postpre_df = self._generate_artificial_pairs(self.n_postpre_pairs, mode='postpre')
        self.prepre_df = self._generate_artificial_pairs(self.n_prepre_pairs, mode='prepre')
        self.df = pd.concat([self.df.iloc[:self.original_len], self.postpost_df, self.postpre_df, self.prepre_df])
    
    def make_subset(self, n_lt_200=2720, n_gt_200=2720, n_artif=960):
        """Make a subset of the dataset with a given number of patches with less
        than 200 landslide pixels, more than 200 landslide pixels, and artificial
        patches"""
        # get indices
        df = self.df.reset_index(drop=True)
        idx_lt_200 = df[df['landslide_pixels'] < 200].index
        idx_gt_200 = df[df['landslide_pixels'] > 200].index
        idx_artif = df[df['landslide_pixels'] == 0].index
        # sample indices
        idx_lt_200 = np.random.choice(idx_lt_200, n_lt_200, replace=False)
        idx_gt_200 = np.random.choice(idx_gt_200, n_gt_200, replace=False)
        idx_artif = np.random.choice(idx_artif, n_artif, replace=False)
        idx = np.concatenate([idx_lt_200, idx_gt_200, idx_artif])
        # create and return subset
        return Subset(self, idx)
    
    def __len__(self):
        return len(self.df)
    
    def _preprocess(self, sample: dict) -> dict:
        """
        Preprocesses post-dem-poly patches

        Args:
            patch (dict): dict containing post, dem, poly patches
        """

        # cast to float32 numpy arrays and transpose (CHW -> HWC)
        sample = {k : v.astype(np.float32).transpose(1,2,0) for k,v in sample.items()}

        # normalize poly and call it 'mask'
        mask = sample.pop('poly') / 255.

        # ignore mask pixels whose value is 1 (i.e. poly's nodata pixels)
        mask[(mask > 0.) & (mask < 1.)] = -1

        # ignore mask pixels whose corresponding post_cm pixels are in ignore_cloud_mask_classes
        post_cm = sample.pop('post_cm')
        #mask = np.where(np.isin(post_cm, self.ignore_cloud_mask_classes), -1, mask)
        mapped_post_cm = np.vectorize(self.cloud_cover_to_mask.get)(post_cm).astype('float32')
        mask = np.where(mapped_post_cm == 1, mask, mapped_post_cm)

        if 'pre' in self.prods:
            # ignore poly pixels whose corresponding pre_cm pixels are in ignore_cloud_mask_classes
            pre_cm = sample.pop('pre_cm')
            #mask = np.where(np.isin(pre_cm, self.ignore_cloud_mask_classes), -1, mask)
            mapped_pre_cm = np.vectorize(self.cloud_cover_to_mask.get)(pre_cm).astype('float32')
            mask = np.where(mapped_pre_cm == 1, mask, mapped_pre_cm)

        # since albumentations needs at least 'image' and 'mask' keys, rename
        # 'post' to 'image' (in _postprocess it will be renamed back to 'post')
        sample['image'] = sample.pop('post')
        sample['mask'] = mask

        return sample
    
    def _augment(self, sample: dict) -> dict:

        # augment common (geometry-disrupting)
        sample = self.augment_common(**sample)

        # get aug parameters (needed in _postprocess to fix aspect after the eventual horizontal flip and random rotation)
        aug_common_replay = {'rot':0, 'hor_flip':False}
        for t in sample['replay']['transforms']:
            if t['__class_fullname__'] == 'RandomRotate90':
                if t['params'] is not None:
                    aug_common_replay['rot'] = t['params']['factor']
            if t['__class_fullname__'] == 'HorizontalFlip':
                aug_common_replay['hor_flip'] = t['applied']
        sample.pop("replay", None)  # remove replay key

        # augment individual (geometry-preserving)
        sample['image'] = self.augment_individual(image=sample['image'])['image']   # post
        if 'pre' in self.prods:
            sample['pre'] = self.augment_individual(image=sample['pre'])['image']   # pre

        return sample, aug_common_replay

    def _postprocess(self, sample: dict, aug_common_replay: dict) -> dict:

        # rename 'image' back to 'post'
        sample['post'] = sample.pop('image')

        # normalize all bands of post images (SSL4EO-S12 preprocessing)
        sample['post'] = np.clip(sample['post'] / 10000., 0., 1.)

        # normalize all bands of pre images (SSL4EO-S12 preprocessing)
        if 'pre' in self.prods:
            sample['pre'] = np.clip(sample['pre'] / 10000., 0., 1.)

        # normalize dem
        sample['dem'] = sample['dem'] / 3000.

        # normalize slope
        sample['slope'] = sample['slope'] / 90. # [0,90] -> [0,1]
        
        # fix aspect after the eventual horizontal flip and random rotation
        sample['aspect']    # is in [0,360]
        if aug_common_replay['hor_flip']:
            sample['aspect'] = 360 - sample['aspect']
        sample['aspect'] = (sample['aspect'] - 90 * aug_common_replay['rot']) % 360 # NB: rotation is counter-clockwise

        # normalize aspect
        #sample['aspect'] = sample['aspect'] / 360.  # [0,360] -> [0,1]
        # decompose aspect in its sin and cos components (by def between [-1,1])
        sample['aspect'] = np.concatenate([
            np.sin(sample['aspect'] * np.pi / 360),
            np.cos(sample['aspect'] * np.pi / 360),
            ], axis=-1)
        # normalize aspect
        sample['aspect'] = (sample['aspect'] + 1) / 2.  # [-1,1] -> [0,1]

        # eventually one-hot encode mask (HW1 -> HW2)
        if self.ohe_mask:
            _mask = np.zeros(sample['mask'].shape[:2] + (2,))
            _mask = np.where(sample['mask'] == -1, [-1.,-1.], _mask)
            _mask = np.where(sample['mask'] == 0, [1.,0.], _mask)
            _mask = np.where(sample['mask'] == 1, [0.,1.], _mask)
            sample['mask'] = _mask.astype(np.float32)

        # cast to torch tensors and transpose (HWC -> CHW)
        to_tensor = lambda arr: torch.from_numpy(arr.transpose(2,0,1))
        sample = {k : to_tensor(v) for k,v in sample.items()}

        
        return sample
    
    def __getitem__(self, idx):

        # get sample
        sample = self.read(idx)
        
        # if postpre/postpost/prepre sample (identified by landslide_pixels == 0), zero out landslide pixels
        if self.df.iloc[idx]['landslide_pixels'] == 0:
            sample['poly'] = np.zeros_like(sample['poly'])
        
        # preprocess sample
        sample = self._preprocess(sample)

        # augment sample
        sample, _aug_common_replay = self._augment(sample)
        
        # postprocess sample
        sample = self._postprocess(sample, _aug_common_replay)

        return sample
    
    def read(self, idx):
        """Reads idx-th patch as a CHW numpy array"""
        
        if self.prefetch:
            sample = self._read_prefetched(idx)

        else:
            sample = {}

            for prod in self.prods:

                # read patch couple
                path = os.path.join(self.root, self.df.iloc[idx][f'fn_{prod}'])
                window = self.df.iloc[idx][['h_start','h_end','w_start','w_end']].to_numpy()

                if prod in ['pre','post']:
                    bands = self.bands
                    raster = read_raster(path, window=window, bands=bands)
                    cm = raster[[-1]]
                    raster = raster[:-1]
                    sample[prod] = raster
                    sample[f'{prod}_cm'] = cm
                else:
                    raster = read_raster(path, window=window)
                    sample[prod] = raster

        return sample
    
    def _read_prefetched(self, idx):
        """Reads idx-th patches from the images cache"""
        sample = {
            'post'    : self._cache[self.df.iloc[idx]['post_patch_id']],
            'post_cm' : self._cache[self.df.iloc[idx]['post_cm_patch_id']],
            'dem'     : self._cache[self.df.iloc[idx]['dem_patch_id']],
            'slope'   : self._cache[self.df.iloc[idx]['slope_patch_id']],
            'aspect'  : self._cache[self.df.iloc[idx]['aspect_patch_id']],
            'poly'    : self._cache[self.df.iloc[idx]['poly_patch_id']],
        }
        if 'pre' in self.prods:
            sample.update({
                'pre' : self._cache[self.df.iloc[idx]['pre_patch_id']],
                'pre_cm' : self._cache[self.df.iloc[idx]['pre_cm_patch_id']],
            })
        
        return sample
    
    def _generate_artificial_pairs(self, n, mode):
        assert mode in ['postpre','postpost','prepre']

        original_df = self.df.iloc[:self.original_len]

        if isinstance(n, float):
            # in this case n is a fraction of len(self.df)
            n = int(n * len(original_df))

        if mode == 'postpre':
            sub_df = original_df.sample(n)
            sub_df['fn_pre'], sub_df['fn_post'] = sub_df['fn_post'], sub_df['fn_pre']   # magic: pre become post and viceversa
            sub_df['landslide_pixels'] = 0 # postpre samples have no landslide pixels (for the sake of change detection)
            if self.prefetch:
                sub_df['pre_patch_id'], sub_df['post_patch_id'] = sub_df['post_patch_id'], sub_df['pre_patch_id']
                sub_df['pre_cm_patch_id'], sub_df['post_cm_patch_id'] = sub_df['post_cm_patch_id'], sub_df['pre_cm_patch_id']
                
            return sub_df
        
        else:
            new_samples = []
            while len(new_samples) < n:
                # select a window
                h_start, h_end, w_start, w_end = original_df.sample(1)[['h_start', 'h_end', 'w_start', 'w_end']].values[0]

                # select all rows referring to that window
                sub_df = self.df[(original_df['h_start'] == h_start) & (original_df['h_end'] == h_end) & (original_df['w_start'] == w_start) & (original_df['w_end'] == w_end)]

                # drop rows with duplicate fn_pre or fn_post
                if mode == 'postpost':
                    sub_df = sub_df.drop_duplicates(subset=['fn_post'])
                elif mode == 'prepre':
                    sub_df = sub_df.drop_duplicates(subset=['fn_pre'])

                if len(sub_df) >= 2:
                    sub_df = sub_df.sample(2)   # sample 2 pre or 2 post disaster patches referring to the same window
                    sample = sub_df.iloc[0]
                    if mode == 'postpost':
                        sample['fn_pre'] = sub_df.iloc[1]['fn_post']    # magic: a post becomes a pre
                        if self.prefetch:
                            sample['pre_patch_id'] = sub_df.iloc[1]['post_patch_id']
                            sample['pre_cm_patch_id'] = sub_df.iloc[1]['post_cm_patch_id']
                    elif mode == 'prepre':
                        sample['fn_post'] = sub_df.iloc[1]['fn_pre']    # magic: a pre becomes a post
                        if self.prefetch:
                            sample['post_patch_id'] = sub_df.iloc[1]['pre_patch_id']
                            sample['post_cm_patch_id'] = sub_df.iloc[1]['pre_cm_patch_id']
                    sample['landslide_pixels'] = 0 # prepre or postpost samples have no landslide pixels (for the sake of change detection)
                    new_samples.append(sample)
                else:
                    continue

        return pd.DataFrame(new_samples)
import os
import numpy as np
import torch
import xarray as xr
import rasterio as rio
import rioxarray as rxr
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix, F1Score, Recall, Precision, JaccardIndex



def validate(loader, model, hparams, criterion=None, threshold=None, verbose=False):
    """
    Returns some metrics (avg. loss, IoU, Dice, precision, recall) for the
    images in the input dataloader, with predictions made by the input model.
    NB: the confusion matrix (and hence IoU, Dice, precision and recall) is
    computed "globally", as if the dataset was a single image.
    """

    hparams = vars(hparams)

    if hparams['N_CLASSES'] == 1:
        task = 'binary'
    elif hparams['N_CLASSES'] > 1 and not hparams['MULTILABEL']:
        task = 'multiclass'
    elif hparams['N_CLASSES'] > 1 and hparams['MULTILABEL']:
        task = 'multilabel'

    # initialize metrics
    # NB: preds inside [0,1] are considered as confidence scores, outside [0,1]
    # as logits. Logits are implicitly converted to confidence scores by the
    # metrics (with sigmoid for 'binary' and 'multilabel', and softmax for
    # 'multiclass'). Additionally, confidence scores are implicitly converted to
    # int tensor with thresholding (binary, multilabel) or argmaxing
    # (multiclass).
    # I explicitly apply the sigmoid/softmax to the logits because we are not
    # sure that they are always outside [0,1]...
    metrics = {
        'cm': ConfusionMatrix(
            task=task,
            num_classes=hparams['N_CLASSES'],   # used if task='multiclass', ignored otherwise
            num_labels=hparams['N_CLASSES'],    # used if task='multilabel', ignored otherwise
            threshold=threshold or hparams['THRESHOLD'],     # used if task='binary' or 'multilabel', ignored otherwise
            ignore_index=-1,
            ).to(hparams['GPU_ID']),
        'f1': F1Score(  # a.k.a. Dice
            task=task,
            num_classes=hparams['N_CLASSES'],   # used if task='multiclass', ignored otherwise
            num_labels=hparams['N_CLASSES'],    # used if task='multilabel', ignored otherwise
            threshold=threshold or hparams['THRESHOLD'],     # used if task='binary' or 'multilabel', ignored otherwise
            average=None,                       # one of ['micro','macro',None]; None -> per-class scores
            multidim_average='global',          # compute score globally (i.e. as if the dataset was a single image)
            ignore_index=-1,
            ).to(hparams['GPU_ID']),
        'recall': Recall(
            task=task,
            num_classes=hparams['N_CLASSES'],   # used if task='multiclass', ignored otherwise
            num_labels=hparams['N_CLASSES'],    # used if task='multilabel', ignored otherwise
            threshold=threshold or hparams['THRESHOLD'],     # used if task='binary' or 'multilabel', ignored otherwise
            average=None,                       # one of ['micro','macro',None]; None -> per-class scores
            multidim_average='global',          # compute score globally (i.e. as if the dataset was a single image)
            ignore_index=-1,
            ).to(hparams['GPU_ID']),
        'precision': Precision(
            task=task,
            num_classes=hparams['N_CLASSES'],   # used if task='multiclass', ignored otherwise
            num_labels=hparams['N_CLASSES'],    # used if task='multilabel', ignored otherwise
            threshold=threshold or hparams['THRESHOLD'],     # used if task='binary' or 'multilabel', ignored otherwise
            average=None,                       # one of ['micro','macro',None]; None -> per-class scores
            multidim_average='global',          # compute score globally (i.e. as if the dataset was a single image)
            ignore_index=-1,
            ).to(hparams['GPU_ID']),
        'iou': JaccardIndex(    # NB: no multidim_average argument
            task=task,
            num_classes=hparams['N_CLASSES'],   # used if task='multiclass', ignored otherwise
            num_labels=hparams['N_CLASSES'],    # used if task='multilabel', ignored otherwise
            threshold=threshold or hparams['THRESHOLD'],     # used if task='binary' or 'multilabel', ignored otherwise
            average=None,                       # one of ['micro','macro',None]; None -> per-class scores
            ignore_index=-1,
            ).to(hparams['GPU_ID']),
    }

    if criterion is not None:
        avg_loss = 0

    model.eval()
    with torch.no_grad():
        for sample in tqdm(loader, disable=not verbose):

            # send sample to GPU
            pre = sample['pre'].to(hparams['GPU_ID'])       # (B, bands, H, W)
            post = sample['post'].to(hparams['GPU_ID'])     # (B, bands, H, W)
            dem = torch.cat([sample['dem'], sample['slope'], sample['aspect']], dim=1).to(hparams['GPU_ID'])   # (B, 4, H, W)
            mask = sample['mask'].to(hparams['GPU_ID'])     # (B, 1, H, W) binary(==categorical) if 'binary', (B, 1, H, W) categorical if 'multiclass', (B, N, H, W) binary if 'multilabel'
            mask = mask.int()   # convert to int tensor (required by torchmetrics)

            # perform a forward pass
            #logits = model(pre, post)
            logits = model(pre, post, dem)    # (B, 1, H, W) logits if 'binary', (B, N, H, W) logits if 'multiclass' and 'multilabel'

            # convert logits to confidence scores in [0,1]
            if hparams['N_CLASSES'] == 1 or hparams['MULTILABEL']:
                pred = F.logsigmoid(logits).exp()    # (B, 1, H, W) in [0,1] if 'binary', (B, N, H, W) in [0,1] if 'multilabel'
            else:
                pred = F.log_softmax(logits, dim=1).exp()    # (B, N, H, W) in [0,1] if 'multiclass'
            
            # squeeze pred and mask if necessary
            logits = logits.squeeze(dim=1)  # becomes (B, H, W) in [0,1] if 'binary', stays (B, N, H, W) in [0,1] if 'multiclass' or 'multilabel'
            pred = pred.squeeze(dim=1)  # becomes (B, H, W) in [0,1] if 'binary', stays (B, N, H, W) in [0,1] if 'multiclass' or 'multilabel'
            mask = mask.squeeze(dim=1)  # becomes (B, H, W) in [0,1] if 'binary' or 'multiclass', stays (B, N, H, W) in [0,1] if 'multilabel'

            # update metrics states
            # NB: this is where thresholding/argmaxing is implicitly performed
            for m_f in metrics.values():
                m_f.update(pred, mask)

            # compute batch loss and update total loss (i.e. reduction='sum')
            # (multiply by batch size to account for the fact that the loss is averaged over the batch)
            if criterion is not None:
                loss = criterion(logits, mask).item() * hparams['BATCH_SIZE']   # inputs are logits, not confidence scores; criterion performs the softmax internally
                avg_loss += loss
    
    # compute metrics
    metrics = {m: m_f.compute() for m, m_f in metrics.items()}

    # compute average loss from total loss
    # (divide by the number of images in the dataset to get the average loss per image)
    if criterion is not None:
        avg_loss = avg_loss / len(loader.dataset)
        metrics['avg_loss'] = avg_loss

    return metrics

# NB: torchmetrics works in this way
# binary --> preds (BHW), target (BHW) (but works in general if preds and target have the same shape, e.g. B1HW)
# multiclass --> preds (BCHW), target (BHW) (doesn't work if target is B1HW!!!)
# multilabel --> preds (BCHW) categorical int/float, target (BCHW) categorical int



def validate_scores(scores, gt_path, threshold=0.5, gpu_id=None):
    """ Compute metrics from confidence scores and ground truth, at a given threshold"""

    assert isinstance(gt_path, str), f"Invalid gt_path {gt_path}. Must be a string."

    metrics = {
        'cm': ConfusionMatrix(
            task='binary',
            threshold=threshold,     # used if task='binary' or 'multilabel', ignored otherwise
            ignore_index=-1,
            ).to(gpu_id),
        'f1': F1Score(  # a.k.a. Dice
            task='binary',
            threshold=threshold,     # used if task='binary' or 'multilabel', ignored otherwise
            average=None,                       # one of ['micro','macro',None]; None -> per-class scores
            multidim_average='global',          # compute score globally (i.e. as if the dataset was a single image)
            ignore_index=-1,
            ).to(gpu_id),
        'recall': Recall(
            task='binary',
            threshold=threshold,     # used if task='binary' or 'multilabel', ignored otherwise
            average=None,                       # one of ['micro','macro',None]; None -> per-class scores
            multidim_average='global',          # compute score globally (i.e. as if the dataset was a single image)
            ignore_index=-1,
            ).to(gpu_id),
        'precision': Precision(
            task='binary',
            threshold=threshold,     # used if task='binary' or 'multilabel', ignored otherwise
            average=None,                       # one of ['micro','macro',None]; None -> per-class scores
            multidim_average='global',          # compute score globally (i.e. as if the dataset was a single image)
            ignore_index=-1,
            ).to(gpu_id),
        'iou': JaccardIndex(    # NB: no multidim_average argument
            task='binary',
            threshold=threshold,     # used if task='binary' or 'multilabel', ignored otherwise
            average=None,                       # one of ['micro','macro',None]; None -> per-class scores
            ignore_index=-1,
            ).to(gpu_id)
    }

    # load gt
    gt = rio.open(gt_path).read() / 255
    # carry invalid pixels (-1) in scores to gt
    scores[gt == -1] = -1
    # convert scores and gt to torch tensor float32
    scores = torch.as_tensor(scores).float().to(gpu_id)
    gt = torch.as_tensor(gt).float().to(gpu_id)

    # update metrics states
    # NB: this is where thresholding/argmaxing is implicitly performed
    for m_f in metrics.values():
        m_f.update(scores, gt)
    
    # compute metrics
    metrics = {m: m_f.compute() for m, m_f in metrics.items()}

    # add threshold to metrics
    metrics['threshold'] = threshold

    return metrics



def predict_scores(model, pre, post, hparams, image_size=None, stride=None, threshold=0.5, verbose=False):
    """
    Predicts the confidence scores for a pre-post image pair using the specified model.
    """

    hparams = vars(hparams)
    
    assert isinstance(pre, str) and isinstance(post, str)
    # open pre and post images
    _pre_path, _post_path = pre, post
    dem = os.path.dirname(os.path.dirname(_pre_path)) + '/dem.tif'
    slope = os.path.dirname(os.path.dirname(_pre_path)) + '/slope.tif'
    aspect = os.path.dirname(os.path.dirname(_pre_path)) + '/aspect.tif'
    pre = rxr.open_rasterio(pre)
    post = rxr.open_rasterio(post)
    dem = rxr.open_rasterio(dem)
    slope = rxr.open_rasterio(slope)
    aspect = rxr.open_rasterio(aspect)

    assert pre.ndim == 3 and post.ndim == 3, f"Input arrays must have 3 dimensions. Got {pre.ndim} for pre and {post.ndim} for post."
    assert pre.shape == post.shape, f"Input arrays must have the same shape. Got {pre.shape} for pre and {post.shape} for post."

    # assign band names
    pre = pre.assign_coords({'band' : list(pre.long_name)})
    post = post.assign_coords({'band' : list(post.long_name)})
    # get mask of nodata pixels
    nodata_mask = pre.sel(band=['B04']).values == 0
    # get pre-post merged thick and thin cloud mask
    pre_cm = np.isin(pre.sel(band=['cloud_mask']).values, [1,2])
    post_cm = np.isin(post.sel(band=['cloud_mask']).values, [1,2])
    cm = np.logical_or(pre_cm, post_cm)
    # select bands and cast them to float32
    pre = pre.sel(band=hparams['BANDS']).astype('float32')
    post = post.sel(band=hparams['BANDS']).astype('float32')
    dem = dem.sel(band=[1]).astype('float32')
    slope = slope.sel(band=[1]).astype('float32')
    aspect = aspect.sel(band=[1]).astype('float32')
    # to torch tensor and send to GPU
    pre = torch.as_tensor(pre.values).to(hparams['GPU_ID']) # torch.as_tensor tries to avoid copying data, if possible (https://stackoverflow.com/a/54260202)
    post = torch.as_tensor(post.values).to(hparams['GPU_ID'])
    dem = torch.as_tensor(dem.values).to(hparams['GPU_ID'])
    slope = torch.as_tensor(slope.values).to(hparams['GPU_ID'])
    aspect = torch.as_tensor(aspect.values).to(hparams['GPU_ID'])
    # preprocessing (like in SSL4EO-S12)
    pre = torch.clip(pre / 10000., min=0., max=1.)
    post = torch.clip(post / 10000., min=0., max=1.)
    dem = dem / 3000.
    slope = slope / 90.
    aspect = torch.cat([
        torch.sin(aspect * torch.pi / 360),
        torch.cos(aspect * torch.pi / 360),
        ], dim=0)
    aspect = (aspect + 1) / 2.  # [-1,1] -> [0,1]
    dem = torch.cat([dem, slope, aspect], dim=0)
    # pad images
    h, w = pre.shape[-2:]
    pad_t, pad_b, pad_l, pad_r = minimum_needed_padding((h,w), hparams['IMAGE_SIZE'], hparams['STRIDE'])
    pre = torch.nn.functional.pad(pre, (pad_l, pad_r, pad_t, pad_b), mode = "reflect")
    post = torch.nn.functional.pad(post, (pad_l, pad_r, pad_t, pad_b), mode = "reflect")
    dem = torch.nn.functional.pad(dem, (pad_l, pad_r, pad_t, pad_b), mode = "reflect")
    slope = torch.nn.functional.pad(slope, (pad_l, pad_r, pad_t, pad_b), mode = "reflect")
    aspect = torch.nn.functional.pad(aspect, (pad_l, pad_r, pad_t, pad_b), mode = "reflect")
    # extract patches
    #pre_patches, patches_idx = extract_patches(pre, hparams['IMAGE_SIZE'], hparams['STRIDE'])
    #post_patches, _ = extract_patches(post, hparams['IMAGE_SIZE'], hparams['STRIDE'])
    # extract patches idx
    image_size = image_size or hparams['IMAGE_SIZE']
    stride = stride or hparams['STRIDE']
    patches_idx = extract_patches_idx(pre.shape[-2:], image_size, stride)
    orig_patches_idx = extract_patches_idx(pre.shape[-2:], 256, 256)    # for merging patches in the case of upsampling
    # perform prediction and get confidence scores
    scores_patches = []
    scores_patches_idx = []
    model.eval()
    with torch.no_grad():
        #for pre_patch, post_patch in tqdm(zip(pre_patches, post_patches), disable=not verbose, desc="Predicting..."):
        for p_idx, orig_p_idx in tqdm(zip(patches_idx, orig_patches_idx), disable=not verbose, desc="Predicting..."):
            t, b, l, r = p_idx
            pre_patch = pre[:, t:b, l:r].unsqueeze(0)
            post_patch = post[:, t:b, l:r].unsqueeze(0)
            dem_patch = dem[:, t:b, l:r].unsqueeze(0)
            if torch.all(pre_patch == 0) or torch.all(post_patch == 0):
                continue    # skip patches with only zeros
            # eventually interpolate to 256
            if image_size != 256:
                pre_patch = F.interpolate(pre_patch, size=256, mode='bilinear', align_corners=False)
                post_patch = F.interpolate(post_patch, size=256, mode='bilinear', align_corners=False)
                dem_patch = F.interpolate(dem_patch, size=256, mode='bilinear', align_corners=False)
            logits = model(pre_patch, post_patch, dem_patch)
            scores = torch.sigmoid(logits).squeeze(dim=0).cpu().numpy()  # (1, 256, 256)
            scores_patches.append(scores)
            scores_patches_idx.append(orig_p_idx)
    # reconstruct whole score map (NB: as numpy array)
    scores = merge_patches(scores_patches, scores_patches_idx, size=pre.shape[-2:])
    # threshold scores
    #out = (scores > threshold).astype('int')
    out = scores
    # remove padding
    out = out[:, pad_t:pad_t+h, pad_l:pad_l+w]
    # zero out nodata pixels
    out[nodata_mask] = -1
    # -1 for thick and thin cloud pixels, and invalid pixels
    out[cm] = -1

    return out



def minimum_needed_padding(img_size, patch_size: int, stride: int):
    """
    Compute the minimum padding needed to make an image divisible by a patch size with a given stride.
    Args:
        image_shape (tuple): the shape (H,W) of the image tensor
        patch_size (int): the size of the patches to extract
        stride (int): the stride to use when extracting patches
    Returns:
        tuple: the padding needed to make the image tensor divisible by the patch size with the given stride
    """
    
    img_size = np.array(img_size)
    pad = np.where(
        img_size <= patch_size,
        (patch_size - img_size) % patch_size,   # the % patch_size is to handle the case img_size = (0,0)
        (stride - (img_size - patch_size)) % stride
        )
    pad_t, pad_l = pad // 2
    pad_b, pad_r = pad[0] - pad_t, pad[1] - pad_l

    return pad_t, pad_b, pad_l, pad_r



def extract_patches(image, patch_size: int, stride: int, order='CHW'):
    """
    Extracts patches from an image tensor.
    Args:
        image (torch.tensor or numpy.ndarray): the image tensor from which to extract patches
        patch_size (int): the size of the patches to extract
        stride (int): the stride to use when extracting patches
    Returns:
        patches: the patches extracted from the image tensor
        patches_idx: the indices of the patches in the original image tensor
    """
    # get image dimensions
    h, w = image.shape[-2:]

    # get the number of patches in each dimension
    n_patches_h = (h - patch_size) // stride + 1
    n_patches_w = (w - patch_size) // stride + 1

    # initialize the patches and patches_idx lists
    patches = []
    patches_idx = []

    # extract patches
    for i in range(n_patches_h):    # iterate over height
        for j in range(n_patches_w):    # iterate over width

            # get the current patch indices
            p_idx = (i*stride, i*stride+patch_size, j*stride, j*stride+patch_size)  # (top, bottom, left, right)

            # get the current patch
            if order == 'CHW':
                p = image[:, p_idx[0]:p_idx[1], p_idx[2]:p_idx[3]]  # (1, 256, 256)
            elif order == 'HWC':
                p = image[p_idx[0]:p_idx[1], p_idx[2]:p_idx[3], :]  # (256, 256, 1)
            
            # append the patch and its indices to the lists
            patches.append(p)
            patches_idx.append(p_idx)

    return patches, patches_idx



def extract_patches_idx(image_size, patch_size: int, stride: int):
    """
    Extracts the indices of the patches in an image tensor.
    Args:
        image_size (tuple): the size (H,W) of the image tensor
        patch_size (int): the size of the patches to extract
        stride (int): the stride to use when extracting patches
    Returns:
        patches_idx: the indices of the patches in the original image tensor
    """
    # get image dimensions
    h, w = image_size

    # get the number of patches in each dimension
    n_patches_h = (h - patch_size) // stride + 1
    n_patches_w = (w - patch_size) // stride + 1

    # initialize the patches_idx list
    patches_idx = []

    # extract patches indices
    for i in range(n_patches_h):    # iterate over height
        for j in range(n_patches_w):    # iterate over width

            # get the current patch indices
            p_idx = (i*stride, i*stride+patch_size, j*stride, j*stride+patch_size)  # (top, bottom, left, right)
            
            # append the patch indices to the list
            patches_idx.append(p_idx)

    return patches_idx



def merge_patches(patches, patches_idx, size=None, order='CHW'):
    """Merge patches into a single image by averaging overlapping pixels"""
    _bottom = patches_idx[-1][1]
    _right = patches_idx[-1][3]
    if size is not None:
        assert len(size) in [2,3] and size[0] >= _bottom and size[1] >= _right, f"Invalid size {size}. Must be at least {_bottom}x{_right}."
    
    H = size[0] if size is not None else _bottom
    W = size[1] if size is not None else _right
    C = patches[0].shape[0] if order == 'CHW' else patches[0].shape[-1]
    merged_img = np.zeros((C,H,W)) if order == 'CHW' else np.zeros((H,W,C))
    n_overlapping_patches = np.zeros((1,H,W)) if order == 'CHW' else np.zeros((H,W,1))

    # loop over patches
    for p, p_idx in zip(patches, patches_idx):
        p = np.array(p)
        t,b,l,r = p_idx
        if order == 'CHW':
            merged_img[:,t:b,l:r] += p
            n_overlapping_patches[:,t:b,l:r] += 1
        else:
            merged_img[t:b,l:r,:] += p
            n_overlapping_patches[t:b,l:r,:] += 1
        
    # compute average
    merged_img = np.divide(merged_img, n_overlapping_patches, where=(n_overlapping_patches != 0))

    return merged_img
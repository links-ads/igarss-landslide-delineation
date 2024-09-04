# %%    IMPORTS
# general
import os
from time import time
from tqdm import tqdm
from loguru import logger
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# data/models/training
import numpy as np
import pandas as pd
import torch
from src.datasets import PatchDataset
from torch.utils.data import DataLoader
from src.model_utils import *
from src.data_utils import *
from src.losses import *
from src.optimizers import *
from src.schedulers import *
from src.arg_parser import parse_args
from src.viz_utils import get_images_for_tb



# %%    HYPERPARAMETERS
hparams = parse_args()

# start tensorboard writer
writer = SummaryWriter(log_dir=f"runs/{os.getenv('SLURM_JOB_ID', default="no_job_id")}")

# %%    LOGGING HYPERPARAMETERS
logger.info("-------- Hyperparameters --------")
logger.info("")
logger.info("--- DATA ---")
logger.info(f"N./frac. of post-pre pairs: {hparams.N_POSTPRE_PAIRS}")
logger.info(f"N./frac. of post-post pairs: {hparams.N_POSTPOST_PAIRS}")
logger.info(f"N./frac. of pre-pre pairs: {hparams.N_PREPRE_PAIRS}")
logger.info(f"Regen. artif. pairs: {hparams.REGEN_ARTIF_PAIRS}")
logger.info(f"Cloud cover to mask: {hparams.CLOUD_COVER_TO_MASK}")
logger.info(f"Bands: {hparams.BANDS}")
logger.info(f"Min. landslide pixels: {hparams.MIN_LANDSLIDE_PIXELS}")
logger.info(f"Max. cloud cover: {hparams.MAX_CLOUD_COVER}")
logger.info("")
logger.info("--- MODEL ---")
logger.info(f"Model: {hparams.MODEL}")
logger.info(f"Encoder (only for unet): {hparams.ENCODER}")
logger.info(f"Fusion module (only for unet): {hparams.FUSION_MODULE}")
logger.info(f"DEM fusion strategy: {hparams.ADD_DEM}")
logger.info(f"Pretrained encoder weights: {hparams.PRETRAINED_ENCODER_WEIGHTS}")
logger.info(f"Threshold on logits (for val): {hparams.THRESHOLD}")
logger.info("")
logger.info("--- TRAINING ---")
logger.info(f"Train on: ['palu', 'mesetas', 'luding', 'iburi']")
logger.info(f"Validate on: ['haiti']")
logger.info(f"Epochs: {hparams.EPOCHS}")
logger.info(f"Batch size: {hparams.BATCH_SIZE}")
logger.info(f"Train on subset: {hparams.TRAIN_ON_SUBSET}")
logger.info(f"Global learning rate: {hparams.LR}")
logger.info(f"Custom learning rate: {hparams.CUSTOM_LR}")
logger.info(f"Global weight decay: {hparams.WEIGHT_DECAY}")
logger.info(f"Custom weight decay: {hparams.CUSTOM_WEIGHT_DECAY}")
logger.info(f"N. classes: {hparams.N_CLASSES}")
logger.info(f"Multilabel: {hparams.MULTILABEL}")
logger.info(f"Loss: {hparams.LOSS}")
logger.info(f"Loss weights: {hparams.LOSS_WEIGHTS}")
if (hparams.LOSS == 'cross_entropy') or ('cross_entropy' in hparams.LOSS):
    logger.info(f"CE class weights: {hparams.CE_CLASS_WEIGHTS}")
    logger.info(f"CE label smoothing: {hparams.CE_LABEL_SMOOTHING}")
if (hparams.LOSS == 'dice') or ('dice' in hparams.LOSS):
    logger.info(f"Dice smooth factor: {hparams.DICE_SMOOTH_FACTOR}")
if (hparams.LOSS == 'focal') or ('focal' in hparams.LOSS):
    logger.info(f"Focal alpha: {hparams.FOCAL_ALPHA}")
    logger.info(f"Focal gamma: {hparams.FOCAL_GAMMA}")
    logger.info(f"Focal normalized: {hparams.FOCAL_NORMALIZED}")
if (hparams.LOSS == 'focal_tversky') or ('focal_tversky' in hparams.LOSS):
    logger.info(f"Focal Tversky alpha: {hparams.FOCAL_TVERSKY_ALPHA}")
    logger.info(f"Focal Tversky beta: {hparams.FOCAL_TVERSKY_BETA}")
    logger.info(f"Focal Tversky gamma: {hparams.FOCAL_TVERSKY_GAMMA}")
if (hparams.LOSS == 'lovasz') or ('lovasz' in hparams.LOSS):
    logger.info(f"Lovasz per image: {hparams.LOVASZ_PER_IMAGE}")
logger.info(f"Early stopping: {hparams.EARLY_STOPPING}")
logger.info("")
logger.info("--- OPTIMIZER ---")
logger.info(f"Optimizer: {hparams.OPTIMIZER}")
logger.info(f"Momentum: {hparams.MOMENTUM}")
logger.info(f"Dampening: {hparams.DAMPENING}")
logger.info(f"Nesterov: {hparams.NESTEROV}")
logger.info(f"AMSGRAD: {hparams.AMSGRAD}")
logger.info(f"Epsilon: {hparams.EPSILON}")
logger.info(f"Betas: {hparams.BETAS}")
logger.info("")
logger.info("--- SCHEDULER ---")
logger.info(f"Scheduler: {hparams.SCHEDULER}")
logger.info(f"ExponentialLR gamma: {hparams.LR_SCHED_EXP_GAMMA}")
logger.info("")
logger.info("--- GENERAL ---")
logger.info(f"Save model as: {hparams.MODEL_NAME}")
logger.info(f"Save model in directory: {hparams.CHECKPOINT_DIR}")
logger.info(f"GPU ID: {hparams.GPU_ID}")
logger.info(f"Number of workers: {hparams.NUM_WORKERS}")
logger.info("")
logger.info("")
logger.info("")



# %%    DATA

# load dataframes with info about patch couples
csv_fn = f'patch_couples_size{hparams.IMAGE_SIZE}_stride{hparams.STRIDE}.csv'
df = {
	'haiti' : pd.read_csv(f'images/nasa/haiti/{csv_fn}'),   # after filtering: 1215 patches
	'palu' : pd.read_csv(f'images/nasa/palu/{csv_fn}'),     # after filtering: 6771 patches
	'mesetas' : pd.read_csv(f'images/mesetas/{csv_fn}'),    # after filtering: 5 patches
	'luding' : pd.read_csv(f'images/luding/{csv_fn}'),      # after filtering: 1360 patches
	'iburi' : pd.read_csv(f'images/iburi/{csv_fn}'),        # after filtering: 7751 patches
	}

train_on = ['palu', 'mesetas', 'luding', 'iburi']
val_on = ['haiti'] # ['iburi']
logger.info(f"Training on: {train_on}")
logger.info(f"Validating on: {val_on}")

train_df = pd.concat([df[inv] for inv in train_on], ignore_index=True)
val_df = pd.concat([df[inv] for inv in val_on], ignore_index=True)

# filter out patch couples with too few visible landslide pixels and too much cloud cover
train_df = train_df[
    #(train_df['landslide_pixels'] > hparams.MIN_LANDSLIDE_PIXELS) & \
    (train_df['cloud_cover'] < hparams.MAX_CLOUD_COVER)
    ]
val_df = val_df[
    (val_df['landslide_pixels'] > hparams.MIN_LANDSLIDE_PIXELS) & \
    (val_df['cloud_cover'] < hparams.MAX_CLOUD_COVER)
    ]
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)



bands = hparams.BANDS + ['cloud_mask']

if hparams.PREFETCH:
    logger.info("Loading data...")

# create datasets
ds = {
    'train' : PatchDataset(
		train_df,
		bands = bands,
		augment_common = PatchDataset.default_train_transforms_common,
		augment_individual = PatchDataset.default_train_transforms_individual,
		prefetch = hparams.PREFETCH,
		verbose = hparams.VERBOSE,
        cloud_cover_to_mask = hparams.CLOUD_COVER_TO_MASK,
        n_postpre_pairs = hparams.N_POSTPRE_PAIRS,
		n_postpost_pairs = hparams.N_POSTPOST_PAIRS,
        n_prepre_pairs = hparams.N_PREPRE_PAIRS,
        ohe_mask = True if hparams.MULTILABEL else False,
		),
    'val' : PatchDataset(
        val_df,
        bands = bands,
        augment_common = PatchDataset.default_eval_transforms_common,
        augment_individual = PatchDataset.default_eval_transforms_individual,
		prefetch = hparams.PREFETCH,
		verbose = hparams.VERBOSE,
        cloud_cover_to_mask = hparams.CLOUD_COVER_TO_MASK,
        ohe_mask = True if hparams.MULTILABEL else False,
        ),
    'test' : PatchDataset(
        test_df,
        bands = bands,
        augment_common = PatchDataset.default_eval_transforms_common,
        augment_individual = PatchDataset.default_eval_transforms_individual,
        prefetch = hparams.PREFETCH,
        verbose = hparams.VERBOSE,
        cloud_cover_to_mask = hparams.CLOUD_COVER_TO_MASK,
        ohe_mask = True if hparams.MULTILABEL else False,
        ),
    'train_infer' : PatchDataset(
        train_df,
        bands = bands,
        augment_common = PatchDataset.default_eval_transforms_common,
        augment_individual = PatchDataset.default_eval_transforms_individual,
		prefetch = False,
		verbose = hparams.VERBOSE,
        cloud_cover_to_mask = hparams.CLOUD_COVER_TO_MASK,
        ohe_mask = True if hparams.MULTILABEL else False,
        ),
    }

# create dataloaders
dl = {
	'train' : DataLoader(ds['train'], batch_size=hparams.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=hparams.NUM_WORKERS, pin_memory=True),
	'val' : DataLoader(ds['val'], batch_size=hparams.BATCH_SIZE, shuffle=False, num_workers=hparams.NUM_WORKERS, pin_memory=True),
    'test' : DataLoader(ds['test'], batch_size=hparams.BATCH_SIZE, shuffle=False, num_workers=hparams.NUM_WORKERS, pin_memory=True),
    }

# log sample images to tensorboard
tb_imgs = get_images_for_tb(ds['train_infer'], 4000, threshold=hparams.THRESHOLD)
writer.add_image("Image/train_pre", tb_imgs['pre'], 0, dataformats='HWC')
writer.add_image("Image/train_post", tb_imgs['post'], 0, dataformats='HWC')
writer.add_image("Image/train_gt", tb_imgs['gt'], 0, dataformats='HWC')
tb_imgs = get_images_for_tb(ds['val'], 998, threshold=hparams.THRESHOLD)
writer.add_image("Image/val_pre", tb_imgs['pre'], 0, dataformats='HWC')
writer.add_image("Image/val_post", tb_imgs['post'], 0, dataformats='HWC')
writer.add_image("Image/val_gt", tb_imgs['gt'], 0, dataformats='HWC')
tb_imgs = get_images_for_tb(ds['test'], 1, threshold=hparams.THRESHOLD)
writer.add_image("Image/test_pre", tb_imgs['pre'], 0, dataformats='HWC')
writer.add_image("Image/test_post", tb_imgs['post'], 0, dataformats='HWC')
writer.add_image("Image/test_gt", tb_imgs['gt'], 0, dataformats='HWC')


# %%    MODELS
logger.info("Loading model...")

# create model
model = make_model(hparams)

# move the model to GPU
model = model.to(hparams.GPU_ID)


# %%    TRAINING
# define loss
criterion = make_loss(hparams)
criterion = criterion.to(hparams.GPU_ID)

# define optimizer
optimizer = make_optimizer(model, hparams)

# define scheduler
scheduler = make_scheduler(optimizer, hparams)	# if hparams['SCHEDULER'] is None, no scheduler is used


# loop over epochs
logger.info("-------- Begin training --------")
best_val_metrics = {'avg_loss': np.inf}
start_time = time()

for epoch in range(hparams.EPOCHS):
    logger.info(f"Epoch {epoch+1}/{hparams.EPOCHS}. Global LR: {scheduler.get_last_lr()[-1] if scheduler is not None else optimizer.param_groups[-1]['lr']:.6f}")

	# set the model in training mode
    model.train()

	# initialize the average training batch loss for the current epoch
    avg_train_loss = 0

    # regenerate artificial postpost, prepre and postpre pairs
    if hparams.REGEN_ARTIF_PAIRS and (epoch > 0): ds['train'].regenerate_artificial_pairs()
    # eventually make a subset of the training set
    if hparams.TRAIN_ON_SUBSET:
        ds_train_sub = ds['train'].make_subset()
        dl['train'] = DataLoader(ds_train_sub, batch_size=hparams.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=hparams.NUM_WORKERS, pin_memory=True)

    # loop over the training set
    for it, sample in enumerate(tqdm(dl['train'], disable=not hparams.VERBOSE)):

        # send sample to GPU
        pre = sample['pre'].to(hparams.GPU_ID)
        post = sample['post'].to(hparams.GPU_ID)
        dem = torch.cat([sample['dem'], sample['slope'], sample['aspect']], dim=1).to(hparams.GPU_ID)
        mask = sample['mask'].to(hparams.GPU_ID)

        # perform a forward pass
        #out = model(pre, post)
        out = model(pre, post, dem)

        # compute the training loss over the minibatch
        loss = criterion(out, mask)
        
        # if the loss is NaN, stop training
        if torch.isnan(loss):
            raise ValueError(f"Training loss is NaN at epoch {epoch+1}, iteration {it+1}. Training stopped.")
        
        # zero out any previously accumulated gradients
        optimizer.zero_grad()

        # compute gradients
        if loss.item() != 0:	# loss is 0 if the batch is all background
            loss.backward()
        else:
            logger.warning(f"Loss is 0 at epoch {epoch+1}, iteration {it+1}. Skipping backward pass.")

        # update weights
        optimizer.step()

        # add the loss to the total training loss so far
        avg_train_loss += loss.item()
        writer.add_scalar("Loss/train (global step)", loss.item(), epoch * len(dl['train']) + it)

        #if it % 1 == 0:
        #    logger.info(f"It: {it+1}, avg training loss: {avg_train_loss / (it+1):.4f}")
	
    # compute average training loss for the current epoch
    avg_train_loss = avg_train_loss / len(dl['train'])
    writer.add_scalar("Loss/train (epoch)", avg_train_loss, epoch+1)

    # update learning rate(s)
    update_lr(scheduler, hparams, metrics=avg_train_loss)
    writer.add_scalar("LR", scheduler.get_last_lr()[-1] if scheduler is not None else optimizer.param_groups[-1]['lr'], epoch + 1)
    
    # log sample prediction on a train image to tensorboard
    tb_imgs = get_images_for_tb(ds['train_infer'], 4000, model, threshold=hparams.THRESHOLD)
    writer.add_image("Image/train_pred", tb_imgs['pred'], epoch+1, dataformats='HWC')

    # validation
    val_metrics = validate(dl['val'], model, hparams, criterion=criterion)



    ######## EPOCH REPORT ########
    logger.info(f"Avg. train loss: {avg_train_loss:.6f}")
    if hparams.N_CLASSES > 1:
        logger.info(f"\nValidation metrics:\n" + \
            "Class:     " + "       ".join([str(i) for i in range(hparams.N_CLASSES)]) + "\n" + \
            "IoU:       " + ", ".join([f"{v:.4f}" for v in val_metrics['iou'].tolist()]) + "\n" + \
            "Dice (F1): " + ", ".join([f"{v:.4f}" for v in val_metrics['f1'].tolist()]) + "\n" + \
            "Precision: " + ", ".join([f"{v:.4f}" for v in val_metrics['precision'].tolist()]) + "\n" + \
            "Recall:    " + ", ".join([f"{v:.4f}" for v in val_metrics['recall'].tolist()]) + "\n" + \
            f"Avg. val. loss: {val_metrics['avg_loss']:.6f}"
        )
    else:
        logger.info(f"\nValidation metrics:\n" + \
            f"IoU: {val_metrics['iou']:.4f}\n" + \
            f"Dice (F1): {val_metrics['f1']:.4f}\n" + \
            f"Precision: {val_metrics['precision']:.4f}\n" + \
            f"Recall: {val_metrics['recall']:.4f}\n" + \
            f"Avg. val. loss: {val_metrics['avg_loss']:.6f}"
        )
    logger.info("")

    writer.add_scalar("Val IoU", val_metrics['iou'][-1].item() if hparams.N_CLASSES > 1 else val_metrics['iou'].item(), epoch + 1)
    writer.add_scalar("Val F1", val_metrics['f1'][-1].item() if hparams.N_CLASSES > 1 else val_metrics['f1'].item(), epoch + 1)
    writer.add_scalar("Val precision", val_metrics['precision'][-1].item() if hparams.N_CLASSES > 1 else val_metrics['precision'].item(), epoch + 1)
    writer.add_scalar("Val recall", val_metrics['recall'][-1].item() if hparams.N_CLASSES > 1 else val_metrics['recall'].item(), epoch + 1)
    writer.add_scalar("Loss/val", val_metrics['avg_loss'], epoch + 1)
    
    tb_imgs = get_images_for_tb(ds['val'], 998, model, threshold=hparams.THRESHOLD)
    writer.add_image("Image/val_pred", tb_imgs['pred'], epoch+1, dataformats='HWC')
    
    # check if the current model achieves the best val loss and eventually save it
    if val_metrics['avg_loss'] < best_val_metrics['avg_loss']:
        best_val_metrics = val_metrics
        ckpt_savepath = os.path.join(hparams.CHECKPOINT_DIR, hparams.MODEL_NAME) + "_best.pt"
        torch.save({
            'state_dict': model.state_dict(),
            'hparams': hparams,
            'val_metrics': val_metrics,
            'epoch': epoch,
            'train_metrics': {'avg_loss': avg_train_loss}
            }, ckpt_savepath)



# save final model
ckpt_savepath = os.path.join(hparams.CHECKPOINT_DIR, hparams.MODEL_NAME) + "_final.pt"
torch.save({
    'state_dict': model.state_dict(),
    'hparams': hparams,
    'val_metrics': val_metrics,
    'epoch': epoch,
    'train_metrics': {'avg_loss': avg_train_loss}
    }, ckpt_savepath)

# log best metrics to tensorboard
writer.add_scalar("Best val IoU", best_val_metrics['iou'][-1].item() if hparams.N_CLASSES > 1 else best_val_metrics['iou'].item(), 0)
writer.add_scalar("Best val F1", best_val_metrics['f1'][-1].item() if hparams.N_CLASSES > 1 else best_val_metrics['f1'].item(), 0)
writer.add_scalar("Best val recall", best_val_metrics['recall'][-1].item() if hparams.N_CLASSES > 1 else best_val_metrics['recall'].item(), 0)
writer.add_scalar("Best val precision", best_val_metrics['precision'][-1].item() if hparams.N_CLASSES > 1 else best_val_metrics['precision'].item(), 0)
writer.add_scalar("Best avg val loss", best_val_metrics['avg_loss'], 0)

# display the total time needed to perform the training
end_time = time()
logger.info("-------- Training ended --------")
logger.info(f"Total time taken to train the model: {end_time-start_time:.2f}s")




# TESTING
logger.info("-------- Begin testing --------")

# load the best model
ckpt = torch.load(ckpt_savepath)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# test
test_metrics = validate(dl['test'], model, hparams, criterion=criterion)

if hparams.N_CLASSES > 1:
    logger.info(f"\nTest metrics:\n" + \
        "Class:     " + "       ".join([str(i) for i in range(hparams.N_CLASSES)]) + "\n" + \
        "IoU:       " + ", ".join([f"{v:.4f}" for v in test_metrics['iou'].tolist()]) + "\n" + \
        "Dice (F1): " + ", ".join([f"{v:.4f}" for v in test_metrics['f1'].tolist()]) + "\n" + \
        "Precision: " + ", ".join([f"{v:.4f}" for v in test_metrics['precision'].tolist()]) + "\n" + \
        "Recall:    " + ", ".join([f"{v:.4f}" for v in test_metrics['recall'].tolist()]) + "\n" + \
        f"Avg. test loss: {test_metrics['avg_loss']:.6f}"
    )
else:
    logger.info(f"\nTest metrics:\n" + \
        f"IoU: {test_metrics['iou']:.4f}\n" + \
        f"Dice (F1): {test_metrics['f1']:.4f}\n" + \
        f"Precision: {test_metrics['precision']:.4f}\n" + \
        f"Recall: {test_metrics['recall']:.4f}\n" + \
        f"Avg. test loss: {test_metrics['avg_loss']:.6f}"
    )

# log test metrics to tensorboard
writer.add_scalar("Test IoU", test_metrics['iou'][-1].item() if hparams.N_CLASSES > 1 else test_metrics['iou'].item(), 0)
writer.add_scalar("Test F1", test_metrics['f1'][-1].item() if hparams.N_CLASSES > 1 else test_metrics['f1'].item(), 0)
writer.add_scalar("Test precision", test_metrics['precision'][-1].item() if hparams.N_CLASSES > 1 else test_metrics['precision'].item(), 0)
writer.add_scalar("Test recall", test_metrics['recall'][-1].item() if hparams.N_CLASSES > 1 else test_metrics['recall'].item(), 0)
writer.add_scalar("Loss/test", test_metrics['avg_loss'], 0)

# log test images to tensorboard
tb_imgs = get_images_for_tb(ds['test'], 1, model, threshold=hparams.THRESHOLD)
writer.add_image("Image/test_pred", tb_imgs['pred'], 0, dataformats='HWC')
writer.close()

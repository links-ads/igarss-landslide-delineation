import argparse
import yaml
import sys
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train a change detection model for landslide delineation')

    # data
    parser.add_argument('--image_size', type=int, default=256, help='Patch size (default: 256)', dest='IMAGE_SIZE')
    parser.add_argument('--stride', type=int, default=128, help='Stride (defaul: 128)', dest='STRIDE')
    #parser.add_argument('--prefetch', type=bool, default=True, help='Prefetch data (default: True)', dest='PREFETCH')
    parser.add_argument('--no_prefetch', action='store_false', help='Do not prefetch data (default: True)', dest='PREFETCH')
    parser.add_argument('--n_postpre_pairs', type=float, default=0, help='Number or fraction of post-pre pairs to generate (default: 0)', dest='N_POSTPRE_PAIRS')
    parser.add_argument('--n_postpost_pairs', type=float, default=0, help='Number or fraction of post-post pairs to generate (default: 0)', dest='N_POSTPOST_PAIRS')
    parser.add_argument('--n_prepre_pairs', type=float, default=0, help='Number or fraction of pre-pre pairs to generate (default: 0)', dest='N_PREPRE_PAIRS')
    parser.add_argument('--regen_artif_pairs', type=bool, default=True, help='Regenerate artificial pairs at each epoch (default: 0)', dest='REGEN_ARTIF_PAIRS')
    parser.add_argument('--cloud_cover_to_mask', type=int, nargs='+', default=[1,-1,-1,1], help='Map each cloud cover class to: 1 = visible, 0 = not visible, -1 = ignore in loss computation. Default: [1,-1,-1,1]', dest='CLOUD_COVER_TO_MASK')
    parser.add_argument('--bands', type=str, nargs='+', default=['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'], help='Bands to use (default: all S2-L2A bands)', dest='BANDS')
    parser.add_argument('--min_landslide_pixels', type=int, default=200, help='Minimum number of landslide pixels for training pairs (default: 200)', dest='MIN_LANDSLIDE_PIXELS')
    parser.add_argument('--max_cloud_cover', type=int, default=0.2, help='Maximum cloud cover for training pairs (default: 0.2)', dest='MAX_CLOUD_COVER')

    # augmentations
    # TODO

    # model
    parser.add_argument('--model', type=str, choices=['unet','bbunet','changeformer','bit','tinycd','seifnet'], default='unet', help='Model architecture (default: unet)', dest='MODEL')
    parser.add_argument('--encoder', type=str, choices=['resnet18', 'resnet50'], default='resnet18', help='Encoder (default: resnet18)', dest='ENCODER')
    parser.add_argument('--fusion_module', type=str, choices=['ssma', 'concat_features', 'diff_features', 'absdiff_features', 'changeformer_diff_features'], default='diff_features', help='Fusion module (default: diff_features)', dest='FUSION_MODULE')
    parser.add_argument('--add_dem', type=str, choices=['early_concat', 'late_concat'], default=None, help='Add DEM to the input with the specified fusion strategy (default: None)', dest='ADD_DEM')
    parser.add_argument('--pretrained_encoder_weights', type=str, default=None, help='Path to pretrained encoder weights (default: None)', dest='PRETRAINED_ENCODER_WEIGHTS')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation (default: 0.5)', dest='THRESHOLD')

    # training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)', dest='EPOCHS')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)', dest='BATCH_SIZE')
    parser.add_argument('--train_on_subset', type=bool, default=True, help='Train on a subset of the training set (6,400 pairs per epoch) (default: True)', dest='TRAIN_ON_SUBSET')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate (default: 1e-3)', dest='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)', dest='WEIGHT_DECAY')
    parser.add_argument('--custom_lr', type=str, default='{}', help='Custom learning rates. Syntax: part_of_model:custom_lr. (default: {})', dest='CUSTOM_LR')
    parser.add_argument('--custom_weight_decay', type=str, default='{}', help='Custom weight decays. Syntax: part_of_model:custom_weight_decay. (default: {})', dest='CUSTOM_WEIGHT_DECAY')
    parser.add_argument('--n_classes', type=int, default=1, choices=[1,2], help='Number of classes for the binary segmentation task (default: 1)', dest='N_CLASSES')
    parser.add_argument('--multilabel', action="store_true", help='Multilabel binary segmentation (default: False)', dest='MULTILABEL')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Enable early stopping (default: False)', dest='EARLY_STOPPING')

    # general
    parser.add_argument('--model_name', type=str, default='unnamed', help='Model name (default: unnamed)', dest='MODEL_NAME')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory (default: checkpoints)', dest='CHECKPOINT_DIR')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use (default: 0)', dest='GPU_ID')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers (default: 4)', dest='NUM_WORKERS')
    parser.add_argument('--yaml', type=str, default=None, help='YAML file containing arguments to override (default: None)', dest='YAML')
    parser.add_argument('--verbose', type=bool, default=False, help='Print tqdm progress bars and other verbose stuff (default: False)', dest='VERBOSE')

    # loss(es)
    parser.add_argument('--loss', type=str, nargs='+', default='cross_entropy', help='Loss function (default: cross_entropy)', dest='LOSS')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1], help='Loss weights (default: [1])', dest='LOSS_WEIGHTS')

    # loss - bce
    parser.add_argument('--ce_class_weights', type=float, nargs='+', default=None, help='Cross-entropy class weights (default: same weight for each class)', dest='CE_CLASS_WEIGHTS')
    parser.add_argument('--ce_label_smoothing', type=float, default=0.0, help='Cross-entropy label smoothing (default: 0)', dest='CE_LABEL_SMOOTHING')

    # loss - dice
    parser.add_argument('--dice_smooth_factor', type=float, default=0.0, help='Smoothness constant for dice coefficient (a) (default: 0.0)', dest='DICE_SMOOTH_FACTOR')

    # loss - focal
    parser.add_argument('--focal_alpha', type=float, default=None, help='Prior probability of having positive value in target (default: None)', dest='FOCAL_ALPHA')
    parser.add_argument('--focal_gamma', type=float, default=2, help='Power factor for dampening weight (focal strength) (default: 2)', dest='FOCAL_GAMMA')
    parser.add_argument('--focal_normalized', type=bool, default=False, help='If True, compute normalized focal loss (default: False)', dest='FOCAL_NORMALIZED')

    # loss - focal_tversky
    parser.add_argument('--focal_tversky_alpha', type=float, default=0.7, help='Focal Tversky alpha (default: 0.7)', dest='FOCAL_TVERSKY_ALPHA')
    parser.add_argument('--focal_tversky_beta', type=float, default=0.3, help='Focal Tversky beta (default: 0.3)', dest='FOCAL_TVERSKY_BETA')
    parser.add_argument('--focal_tversky_gamma', type=float, default=2, help='Focal Tversky gamma (default: 2)', dest='FOCAL_TVERSKY_GAMMA')
    parser.add_argument('--focal_tversky_smooth_factor', type=float, default=0.0, help='Smoothness constant (default: 0.0)', dest='FOCAL_TVERSKY_SMOOTH_FACTOR')

    # loss - lovasz
    parser.add_argument('--lovasz_per_image', type=bool, default=False, help='If True loss computed per each image and then averaged, else computed per whole batch (default: False)', dest='LOVASZ_PER_IMAGE')

    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'radam', 'sgd'], default='adamw', help='Optimizer (default: adamw)', dest='OPTIMIZER')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor (default: 0.9)', dest='MOMENTUM')
    parser.add_argument('--dampening', type=float, default=0.0, help='Dampening for momentum (default: 0)', dest='DAMPENING')
    parser.add_argument('--nesterov', type=bool, default=False, help='Enables Nesterov momentum (default: False)', dest='NESTEROV')
    parser.add_argument('--amsgrad', type=bool, default=True, help='Whether to use the AMSGrad variant of AdamW (default: True)', dest='AMSGRAD')
    parser.add_argument('--eps', type=float, default=1e-8, help='Term added to the denominator to improve numerical stability (default: 1e-8)', dest='EPSILON')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.9, 0.999], help='Coefficients used for computing running averages of gradient and its square (default: [0.9, 0.999])', dest='BETAS')

    # scheduler
    parser.add_argument('--scheduler', type=str, default=None, help='LR Scheduler (default: None)', dest='SCHEDULER')

    # scheduler - plateau
    parser.add_argument('--lr_sched_plateau_patience', type=int, default=5, help='Number of epochs with no improvement after which learning rate will be reduced (default: 5)', dest='LR_SCHED_PLATEAU_PATIENCE')
    parser.add_argument('--lr_sched_plateau_factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced (new_lr = lr * factor) (default: 0.1)', dest='LR_SCHED_PLATEAU_FACTOR')
    parser.add_argument('--lr_sched_plateau_min_lr', type=float, default=1e-7, help='A lower bound on the learning rate of all param groups or each group respectively (default: 1e-7)', dest='LR_SCHED_PLATEAU_MIN_LR')
    parser.add_argument('--lr_sched_plateau_threshold', type=float, default=0.0001, help='Threshold for measuring the new optimum (default: 0.0001)', dest='LR_SCHED_PLATEAU_THRESHOLD')
    parser.add_argument('--lr_sched_plateau_threshold_mode', type=str, choices=['rel', 'abs'], default='rel', help='One of rel, abs (default: rel)', dest='LR_SCHED_PLATEAU_THRESHOLD_MODE')
    parser.add_argument('--lr_sched_plateau_cooldown', type=int, default=0, help='Number of epochs to wait before resuming normal operation after lr has been reduced (default: 0)', dest='LR_SCHED_PLATEAU_COOLDOWN')

    # scheduler - cyclic
    parser.add_argument('--lr_sched_cyclic_base_lr', type=float, default=1e-5, help='Initial learning rate which is the lower boundary in the cycle for each param group (default: 1e-5)', dest='LR_SCHED_CYCLIC_BASE_LR')
    parser.add_argument('--lr_sched_cyclic_max_lr', type=float, default=1e-1, help='Upper learning rate boundaries in the cycle for each parameter group (default: 1e-1)', dest='LR_SCHED_CYCLIC_MAX_LR')
    parser.add_argument('--lr_sched_cyclic_step_size_up', type=int, default=2000, help='Number of training iterations in the increasing half of a cycle (default: 2000)', dest='LR_SCHED_CYCLIC_STEP_SIZE_UP')
    parser.add_argument('--lr_sched_cyclic_step_size_down', type=int, default=None, help='Number of training iterations in the decreasing half of a cycle. If step_size_down is None, it is set to step_size_up (default: None)', dest='LR_SCHED_CYCLIC_STEP_SIZE_DOWN')
    parser.add_argument('--lr_sched_cyclic_mode', type=str, choices=['triangular', 'triangular2', 'exp_range'], default='triangular', help='One of {triangular, triangular2, exp_range} (default: triangular)', dest='LR_SCHED_CYCLIC_MODE')
    parser.add_argument('--lr_sched_cyclic_gamma', type=float, default=1.0, help='Constant in "exp_range" mode: gamma**(cycle iterations) (default: 1.0)', dest='LR_SCHED_CYCLIC_GAMMA')
    parser.add_argument('--lr_sched_cyclic_scale_fn', type=str, choices=['cos', 'sin'], default='cos', help='Scaling function: one of {cos, sin} (default: cos)', dest='LR_SCHED_CYCLIC_SCALE_FN')
    parser.add_argument('--lr_sched_cyclic_scale_mode', type=str, choices=['cycle', 'iterations'], default='cycle', help='One of {cycle, iterations} (default: cycle)', dest='LR_SCHED_CYCLIC_SCALE_MODE')
    parser.add_argument('--lr_sched_cyclic_cycle_momentum', type=bool, default=True, help='Cycle momentum between high and low bounds (default: True)', dest='LR_SCHED_CYCLIC_CYCLE_MOMENTUM')
    parser.add_argument('--lr_sched_cyclic_base_momentum', type=float, default=0.8, help='Lower momentum boundaries in the cycle for each parameter group (default: 0.8)', dest='LR_SCHED_CYCLIC_BASE_MOMENTUM')
    parser.add_argument('--lr_sched_cyclic_max_momentum', type=float, default=0.9, help='Upper momentum boundaries in the cycle for each parameter group (default: 0.9)', dest='LR_SCHED_CYCLIC_MAX_MOMENTUM')
    
    # scheduler - cosine (with warmup)
    # TODO

    # scheduler - onecycle
    # TODO

    # scheduler - exponential
    parser.add_argument('--lr_sched_exp_gamma', type=float, default=0.99, help='Multiplicative factor of learning rate decay (default: 0.99)', dest='LR_SCHED_EXP_GAMMA')
    parser.add_argument('--lr_sched_exp_last_epoch', type=int, default=-1, help='The index of last epoch (default: -1)', dest='LR_SCHED_EXP_LAST_EPOCH')


    # parse arguments
    args = parser.parse_args()  # a Namespace object containing all command-line/default arguments as attributes

    # discriminate between default args and given args
    cmdline_args = {}
    default_args = {}
    argv = [arg[2:].split('=')[0] for arg in sys.argv if arg.startswith('--')]  # argument vector
    for k,v in vars(args).items():
        if k.lower() in argv:
            cmdline_args[k] = v
        else:
            default_args[k] = v
    
    # read yaml file arguments, if provided
    yaml_args = {}
    if args.YAML is not None:
        with open(args.YAML, 'r') as f:
            yaml_args = yaml.safe_load(f)        

    # merge arguments in the following (increasing) order of priority: default < yaml < cmdline
    # arguments with higher priority will override ones with lower priority
    args = argparse.Namespace(**{**default_args, **yaml_args, **cmdline_args})

    return args
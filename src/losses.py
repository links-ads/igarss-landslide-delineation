import numpy as np
import torch
import segmentation_models_pytorch as smp
#DiceLoss, FocalLoss, TverskyLoss, LovaszLoss, SoftBCEWithLogitsLoss

# CORRECTION: gamma parameter of focal tversky loss should be applied per-class, and not after the aggregation of the per-class losses.
# https://github.com/qubvel-org/segmentation_models.pytorch/issues/734
smp.losses.TverskyLoss.aggregate_loss = lambda self, loss: (loss ** self.gamma).mean()


def make_loss(hparams):
    """ Loss factory: make loss from hparams """
    return CustomLoss(hparams)



class CustomLoss(torch.nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        hparams = vars(hparams)
        
        if isinstance(hparams['LOSS'], str):
            hparams['LOSS'] = [hparams['LOSS']]     # convert to list
        self.losses = self._custom_loss(hparams)    # make specified loss combination

        self.weights = hparams.get('LOSS_WEIGHTS', [1]*len(hparams['LOSS']))
        if isinstance(self.weights, (float, int)):
            self.weights = [self.weights]           # convert to list
        
        assert len(self.losses) == len(self.weights), f"The number of losses and weights must be the same. Got {len(self.losses)} losses and {len(self.weights)} weights"
        assert all([w >= 0 for w in self.weights]), f"Weight of each loss must be non-negative. Got weights: {self.weights}"



    def forward(self, logits, targets):

        loss = 0
        for l,w in zip(self.losses, self.weights):
            if isinstance(l, torch.nn.CrossEntropyLoss):
                # CrossEntropyLoss requires BHW target, it does not support B1HW
                loss += w * l(logits, targets.squeeze(1).long())
            else:
                loss += w * l(logits, targets)
        return loss
    


    def _custom_loss(self, hparams):
        # TODO add more losses
        # TODO add support for multi-class segmentation (mode='multiclass')
        available_losses = ['cross_entropy', 'dice', 'focal', 'focal_tversky', 'lovasz']
        assert all([l in available_losses for l in hparams['LOSS']]), f"Available losses are: {available_losses}. Got {hparams['LOSS']}"
        
        # define custom loss function
        losses = []

        if hparams['N_CLASSES'] == 1:
            mode = 'binary'
        elif hparams['N_CLASSES'] > 1 and not hparams['MULTILABEL']:
            mode = 'multiclass'
        elif hparams['N_CLASSES'] > 1 and hparams['MULTILABEL']:
            mode = 'multilabel'
        
        for l in hparams['LOSS']:
            

            if l == 'cross_entropy':

                # get class weights vector
                if hparams['CE_CLASS_WEIGHTS'] is None:
                    class_weights = None
                else:
                    assert len(hparams['CE_CLASS_WEIGHTS']) == hparams['N_CLASSES'], f"If given, CE_CLASS_WEIGHTS must have the same length as N_CLASSES. Got {len(hparams['CE_CLASS_WEIGHTS'])} weights for {hparams['N_CLASSES']} classes"
                    class_weights = torch.tensor(hparams['CE_CLASS_WEIGHTS'], requires_grad=False).float().to(hparams['GPU_ID'])
                    if mode == 'multilabel':
                        class_weights = class_weights.reshape(1,hparams['N_CLASSES'],1,1)
                # get label smoothing value
                label_smoothing = torch.tensor(hparams['CE_LABEL_SMOOTHING'], requires_grad=False).float().to(hparams['GPU_ID'])

                if mode in ['binary', 'multilabel']:
                    criterion = smp.losses.SoftBCEWithLogitsLoss(
                        pos_weight=class_weights,   # for broadcasting logic see: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
                        smooth_factor=label_smoothing,
                        reduction='mean',
                        ignore_index=-1,
                        )
                    # y_pred are BCHW logits (in the 'binary' case, C=1). logits are internally converted to confidence scores in [0,1] with a sigmoid function.
                    # y_true are BCHW binary targets (in the 'binary' case, C=1)
                    # in any case, y_pred and y_true must have the same shape
                    # NB: in smp docs it seems like y_true should be BHW or B1HW (i.e. supports only 'binary'), but it is not true. It can be BCHW because it is based on F.binary_cross_entropy_with_logits
                elif mode == 'multiclass':
                    criterion = torch.nn.CrossEntropyLoss(
                        weight=class_weights,
                        label_smoothing=label_smoothing,
                        reduction='mean',
                        ignore_index=-1,
                        )
                    # y_pred are BCHW logits. logits are internally converted to confidence scores in [0,1] with a softmax function.
                    # y_true are BHW categorical targets (in this case, they will be implicitly one-hot-encoded to shape BCHW) or BCHW one-hot-encoded (or probability) targets
                    # the difference between torch.nn.CrossEntropyLoss and smp.losses.SoftCrossEntropyLoss is that the latter DOES NOT support BCHW targets, but only BHW (so it is worse...)

            elif l == 'dice':
                criterion = smp.losses.DiceLoss(
                    mode=mode,
                    from_logits=True,
                    smooth=hparams['DICE_SMOOTH_FACTOR'],
                    ignore_index=-1,
                    )   # see NB below
            
            elif l == 'focal':
                criterion = smp.losses.FocalLoss(
                    mode=mode,
                    alpha=hparams['FOCAL_ALPHA'],
                    gamma=hparams['FOCAL_GAMMA'],
                    normalized=hparams['FOCAL_NORMALIZED'],
                    ignore_index=-1,
                    )   # see NB below
                

            elif l == 'focal_tversky':
                criterion = smp.losses.TverskyLoss(
                    mode = mode,
                    alpha = hparams['FOCAL_TVERSKY_ALPHA'],
                    beta = hparams['FOCAL_TVERSKY_BETA'],
                    gamma = hparams['FOCAL_TVERSKY_GAMMA'],
                    smooth = hparams['FOCAL_TVERSKY_SMOOTH_FACTOR'],
                    from_logits = True,
                    ignore_index = -1,
                    )   # see NB below
                

            elif l == 'lovasz':
                criterion = smp.losses.LovaszLoss(
                    mode = mode,
                    per_image = hparams['LOVASZ_PER_IMAGE'],
                    ignore_index = -1,
                    from_logits = True,
                    )   # see NB below
                

            else:
                criterion = None
            
            losses.append(criterion)

        return losses
    
        # NB for losses from segmentation_models_pytorch:
        # - if from_logits=True, y_pred must be logits. They are converted to confidence scores in [0,1] with a sigmoid function in the binary and multilabel mode, and with a softmax function in the multiclass mode
        # - if mode='binary', both y_pred and y_true must have shape BHW or B1HW (the values are referred to the positive class)
        # - if mode='multiclass', y_pred must be BCHW and y_true must be BHW or B1HW. y_true are categorical targets (not one-hot-encoded), and they will be implicitly one-hot-encoded to shape BCHW (class-wise boolean targets)
        # SO y_true DOES NOT SUPPORT shape BCHW AS IN TORCHVISION (only classes, not probabilities for each class; infact to work with y_true probabilities we would need the generalized dice score...)
        # - if mode='multilabel', y_pred must be BCHW and y_true must be BCHW. In this case it is assumed that y_true is already one-hot-encoded (class-wise boolean targets)
    



    def __repr__(self):
        return super().__repr__() + f"with loss(es): {self.losses} and loss weight(s): {self.weights}"






"""
NOTA BENE

from segmentation_models_pytorch.losses import SoftCrossEntropyLoss
criterion = SoftCrossEntropyLoss(smooth_factor=0.0, reduction='none')
pred = torch.tensor([[[0,0,0],[0,2.19722,-0.40546]]]).unsqueeze(-1)    # 1x2x3x1
target = torch.tensor([[[0,1,1]]]).unsqueeze(-1).long()    # 1x1x3x1
loss = criterion(pred, target)
print(loss)

from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
criterion = SoftBCEWithLogitsLoss(smooth_factor=0.0, reduction='none')
pred = torch.tensor([[[0,2.19722,-0.40546]]]).unsqueeze(-1)    # 1x1x3x1
target = torch.tensor([[[0,1,1]]]).unsqueeze(-1).float()    # 1x1x3x1
loss = criterion(pred, target)
print(loss)



from torch.nn import CrossEntropyLoss
criterion = CrossEntropyLoss(reduction='none')
pred = torch.tensor([[[0,0,0],[0,2.19722,-0.40546]]]).unsqueeze(-1)    # 1x2x3x1
target = torch.tensor([[[1,0,0],[0,1,1]]]).unsqueeze(-1).float()    # 1x1x3x1
loss = criterion(pred, target)
print(loss)

from torch.nn import BCEWithLogitsLoss
criterion = BCEWithLogitsLoss(reduction='none')
pred = torch.tensor([[[0,2.19722,-0.40546]]]).unsqueeze(-1)    # 1x1x3x1
target = torch.tensor([[[0,1,1]]]).unsqueeze(-1).float()    # 1x1x3x1
loss = criterion(pred, target)
print(loss)

DANNO TUTTE LO STESSO RISULTATO, PERCHé SI DIMOSTRA CHE CROSSENTROPY è BINARYCROSSENTROPY QUANDO IL N. DI CLASSI PER LA CROSSENTROPY è 2.



INVECE ALTRE LOSS TIPO LA DICE CAMBIANO COMPORTAMENTO TRA BINARY E MULTICLASS, E.G.:

from segmentation_models_pytorch.losses import DiceLoss

pred = torch.tensor([[[[1,1],[1,0]]]])
pred2 = torch.cat([1-pred, pred], dim=1)
true = torch.tensor([[[[0,1],[1,0]]]])

criterion = DiceLoss(mode='binary', from_logits=False)
print(criterion(pred, true))
> 0.2

criterion = DiceLoss(mode='multiclass', from_logits=False)
print(criterion(pred2, true))
> 0.266667

LA SCELTA DELLA MODALITà 'binary' O 'multiclass' QUINDI CAMBIA IL GRADIENTE E QUINDI L'ADDESTRAMENTO... PROVA A USARE N_CLASSES = 2 CON LA DICE E LA FOCAL_TVERSKY E VEDI COSA CAMBIA
"""
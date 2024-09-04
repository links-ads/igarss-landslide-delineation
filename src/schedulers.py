import torch
import torch.optim.lr_scheduler



def make_scheduler(optimizer, hparams):
    """ LR scheduler factory: make scheduler from optimizer and hparams """

    hparams = vars(hparams)

    scheduler_name = hparams['SCHEDULER']
    
    if scheduler_name is None:
        return None
    
    assert scheduler_name in ['plateau', 'cosine', 'cyclic', 'onecycle', 'exp'], f"Scheduler must be one of: ['plateau', 'cosine', 'cyclic', 'onecycle']. Got {scheduler_name}"
    
    if scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience = hparams['LR_SCHED_PLATEAU_PATIENCE'],
            factor = hparams['LR_SCHED_PLATEAU_FACTOR'],
            min_lr = hparams['LR_SCHED_PLATEAU_MIN_LR'],
            threshold = hparams['LR_SCHED_PLATEAU_THRESHOLD'],
            threshold_mode = hparams['LR_SCHED_PLATEAU_THRESHOLD_MODE'],
            cooldown = hparams['LR_SCHED_PLATEAU_COOLDOWN'],
            )
        
    elif scheduler_name == 'cosine':
        raise NotImplementedError   # TODO

    elif scheduler_name == 'cyclic':
        raise NotImplementedError   # TODO

    elif scheduler_name == 'onecycle':
        raise NotImplementedError   # TODO
    
    elif scheduler_name == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = hparams['LR_SCHED_EXP_GAMMA'],
            last_epoch = hparams['LR_SCHED_EXP_LAST_EPOCH'],
            )
    
    return scheduler



def update_lr(scheduler, hparams, metrics=None):
    """ Update learning rate(s) according to the specified scheduler """

    hparams = vars(hparams)

    if scheduler is not None:
        scheduler_name = hparams['SCHEDULER']
    
        if scheduler_name == 'plateau':
            assert metrics is not None, 'Plateau scheduler requires metrics to be passed'
            scheduler.step(metrics)
        
        elif scheduler_name == 'cosine':
            raise NotImplementedError   # TODO

        elif scheduler_name == 'cyclic':
            raise NotImplementedError   # TODO

        elif scheduler_name == 'onecycle':
            raise NotImplementedError   # TODO
        
        elif scheduler_name == 'exp':
            scheduler.step()

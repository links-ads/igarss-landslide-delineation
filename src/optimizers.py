import json
import torch
import torch.optim



ACCEPTED_OPTIMIZERS = ['adam', 'adamw', 'radam', 'sgd']



def make_optimizer(model, hparams):
    """ Optimizer factory: make optimizer from model and hparams """

    hparams = vars(hparams)
    
    optimizer_name = hparams['OPTIMIZER']
    assert optimizer_name in ACCEPTED_OPTIMIZERS, f"Optimizer must be one of: {ACCEPTED_OPTIMIZERS}. Got {optimizer_name}"

    # get optim groups of learnable parameters
    optim_groups = get_optim_groups(model)

    # exclude biases and blacklisted layers' parameters from weight decay and add custom weight decays
    # on why: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
    if isinstance(custom_wd := hparams['CUSTOM_WEIGHT_DECAY'], str):
        custom_wd = json.loads(custom_wd)
    optim_groups = _add_custom_wd(optim_groups, custom_wd=custom_wd)

    # add custom learning rates
    if isinstance(custom_lr := hparams['CUSTOM_LR'], str):
        custom_lr = json.loads(custom_lr)
    optim_groups = _add_custom_lr(optim_groups, custom_lr=custom_lr)


    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            optim_groups,
            lr = hparams['LR'],
            weight_decay = hparams['WEIGHT_DECAY'],
            amsgrad = hparams['AMSGRAD'],
            betas = hparams['BETAS'],
            eps = hparams['EPSILON'],
            )
        
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr = hparams['LR'],
            weight_decay = hparams['WEIGHT_DECAY'],
            amsgrad = hparams['AMSGRAD'],
            betas = hparams['BETAS'],
            eps = hparams['EPSILON'],
            )
    
    elif optimizer_name == 'radam':
        optimizer = torch.optim.RAdam(
            optim_groups,
            lr = hparams['LR'],
            weight_decay = hparams['WEIGHT_DECAY'],
            betas = hparams['BETAS'],
            eps = hparams['EPSILON'],
            )
        
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            optim_groups,
            lr = hparams['LR'],
            weight_decay = hparams['WEIGHT_DECAY'],
            momentum = hparams['MOMENTUM'],
            dampening = hparams['DAMPENING'],
            nesterov = hparams['NESTEROV'],
            )
        
    return optimizer



def get_layers(model):
    submodules = list(model.children())
    if len(submodules) == 0:
        return [model]
    else:
        res = []
        for module in submodules:
            res += get_layers(module)
        return res



def get_optim_groups(model):
    # TODO add support for nn.Parameter modules
    # UPDATE: support for nn.Parameter modules added, but only when
    # nn.Parameters are the first to appear in the model.parameters() list,
    # OTHERWISE THE "full_param_name" WILL BE WRONG FOR SOME PARAMETERS!!!

    layers = get_layers(model)

    layer_params = []
    for layer in layers:
        layer_params += list(layer.parameters())

    optim_groups = []

    if len(list(model.parameters())) != len(layer_params):
        # if this is the case, maybe there are some nn.Parameters that are not in any group. This is because they are not in any torch.nn layer. Let's find them

        for n,p in model.named_parameters():
            if not any([p is lp for lp in layer_params]):
                optim_groups.append(
                    {"params" : p, "class" : torch.nn.Parameter, "param_type" : "parameter"}
                )
    
    for layer in layers:
        for param_type, params in layer.named_parameters():  # this skips all layers without parameters (e.g. certain non-parametric fusion modules)
            optim_groups.append(
                {"params" : params, "class" : type(layer), "param_type" : param_type}
            )
    # add full parameter names
    # NB:
    # > all([i['param'] is j[1] for i,j in zip(get_optim_groups(get_layers(model)), model.named_parameters())])
    # True
    for i, (fpn, _) in enumerate(model.named_parameters()):
        optim_groups[i]["full_param_name"] = fpn

    return optim_groups



def _add_custom_wd(optim_groups, custom_wd={}):
    """
    Adds specified weight decay to the specified optimizer groups. Specifically:
    1. excludes bias parameters and blacklisted layers' parameters from weight decay
    2. eventually applies a different weight decay to parts of the model specified in custom_wd
    NB: custom_wd should be a dict, with keys being the parts of the model to
    which a custom weight decay should be applied, and values being the custom
    weight decay to apply.
    Inspired by: https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L263
    """
    blacklist_layers = (torch.nn.modules.batchnorm._NormBase, torch.nn.LayerNorm, torch.nn.Embedding)

    for optim_group in optim_groups:
        if optim_group["class"] in blacklist_layers:
            # don't apply wd to blacklisted layers
            optim_group["weight_decay"] = 0.0
        elif optim_group["param_type"].endswith("bias"):
            # don't apply wd to bias parameters
            optim_group["weight_decay"] = 0.0
        #elif (part := optim_group["full_param_name"].split('.')[0]) in custom_wd.keys():
        #    # apply a custom weight decay to the parameter
        #    optim_group["weight_decay"] = custom_wd[part]
        else:
            for k in custom_wd.keys():
                if optim_group["full_param_name"].startswith(k):
                    # apply a custom weight decay to the parameter
                    optim_group["weight_decay"] = custom_wd[k]
    
    return optim_groups



def _add_custom_lr(optim_groups, custom_lr={}):
    """
    Adds specified learning rate to the specified optimizer groups.
    NB: custom_lr should be a dict, with keys being the parts of the model to
    which a custom learning rate should be applied, and values being the custom
    learning rate to apply.
    Inspired by: https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L263
    """
    for optim_group in optim_groups:
        # remove the last part of the full parameter name (usually '.weight','.bias','.running_mean','.running_var','.num_batches_tracked')
        #part = '.'.join(optim_group["full_param_name"].split('.')[:-1])
        for k in custom_lr.keys():
            #if part.startswith(k):
            if optim_group["full_param_name"].startswith(k):
                if custom_lr[k] > 0.0:
                    # apply a custom learning rate to the parameter
                    optim_group["lr"] = custom_lr[k]
                elif custom_lr[k] == 0.0:
                    # set requires_grad to False (computationally faster than setting lr to 0)
                    optim_group["params"].requires_grad = False
    
    return optim_groups
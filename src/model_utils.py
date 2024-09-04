from loguru import logger
from collections import OrderedDict
import os.path as osp
import torch
from src.models import *
import segmentation_models_pytorch as smp
from src.fusion_modules import *
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder



def make_model(hparams):
    """ Model factory: returns a model instance based on the provided hyperparameters. """

    hparams = vars(hparams)

    # load SSL4EO weights from checkpoint
    if hparams['PRETRAINED_ENCODER_WEIGHTS'] is not None:
        pretrained_encoder_checkpoint = torch.load(hparams['PRETRAINED_ENCODER_WEIGHTS'], map_location=torch.device('cpu'))
        if 'state_dict' in pretrained_encoder_checkpoint:   # MoCo
            pretrained_encoder_weights = pretrained_encoder_checkpoint['state_dict']
        elif 'teacher' in pretrained_encoder_checkpoint:    # DINO
            pretrained_encoder_weights = pretrained_encoder_checkpoint['teacher']
        else:
            raise ValueError('pretrained_encoder_checkpoint must contain either "state_dict" (MoCo pretraining) or "teacher" (DINO pretraining)')
        
        pretrained_encoder_weights = convert_state_dict(pretrained_encoder_checkpoint, bands=hparams['BANDS'])    # to make sure the weights are in the right format (torchvision)

    # instantiate model
    if hparams['MODEL'] == 'unet':

        model = smp.Unet(
            encoder_name = hparams['ENCODER'],
            encoder_weights = None,                 # we will load the pretrained weights later
            in_channels = len(hparams['BANDS']),    # model input channels (12 for Sentinel-2 level 2A)
            classes = hparams['N_CLASSES'],   # model output channels (1 for binary segmentation)
            )
        
        # modify the model's encoder to make it bitemporal
        #model = make_unet_bitemporal(model, fusion_module=hparams['FUSION_MODULE'])
        model = make_unet_bitemporal(model, fusion_module=hparams['FUSION_MODULE'], add_dem=hparams['ADD_DEM'])

    elif hparams['MODEL'] == 'seifnet':
        model = SEIFNet(
            input_nc = len(hparams['BANDS']),
            output_nc = hparams['N_CLASSES'],
            )
    
    elif hparams['MODEL'] == 'bbunet':
        assert hparams['ENCODER'] == 'resnet18', f"Only 'resnet18' encoder is supported for BBUnet. Got {hparams['ENCODER']}"
        model = BBUnet(              # we will load the pretrained weights later
            s2_in_channels = len(hparams['BANDS']),    # model input channels (12 for Sentinel-2 level 2A)
            dem_in_channels = 4,
            classes = hparams['N_CLASSES'],   # model output channels (1 for binary segmentation)
        )
    
    elif hparams['MODEL'] == 'changeformer':
        model = ChangeFormer(
            input_nc = len(hparams['BANDS']),
            output_nc = hparams['N_CLASSES'],
            decoder_softmax = False,
            embed_dim = 256
        )

    elif hparams['MODEL'] == 'bit':
        model = BIT(
            arch = 'base_transformer_pos_s4_dd8',
            input_nc = len(hparams['BANDS']),
            output_nc = hparams['N_CLASSES'],
        )

    elif hparams['MODEL'] == 'tinycd':
        model = TinyCD(
            bkbn_name = "efficientnet_b4",
            input_nc = len(hparams['BANDS']),
            pretrained = True,
            output_layer_bkbn = "3",
            freeze_backbone = False,
        )
        
    else:
        raise NotImplementedError(f"Model {hparams['MODEL']} (still?) not implemented")

    # load pretrained weights
    if hparams['PRETRAINED_ENCODER_WEIGHTS'] is not None:
        mismatched_keys = model.encoder.load_state_dict(pretrained_encoder_weights, strict=False)

        if hparams['VERBOSE']:
            if mismatched_keys is None:
                logger.info(f"Pretrained encoder weights loaded successfully from {hparams['PRETRAINED_ENCODER_WEIGHTS']}")
            else:
                logger.info(f"Missing keys: {mismatched_keys.missing_keys}")
                logger.info(f"Unexpected keys: {mismatched_keys.unexpected_keys}")
    
    # eventually add two channels to the first conv layer for DEM slope and aspect
    if hparams['ADD_DEM'] == 'early_concat':
        new_conv1 = torch.nn.Conv2d(
            in_channels = len(hparams['BANDS'])+2,
            out_channels = model.encoder.conv1.out_channels,
            kernel_size = model.encoder.conv1.kernel_size,
            stride = model.encoder.conv1.stride,
            padding = model.encoder.conv1.padding,
            bias = model.encoder.conv1.bias,
            )
        with torch.no_grad():
            new_conv1.weight[:, :len(hparams['BANDS']), :, :] = model.encoder.conv1.weight
            new_conv1.weight[:, len(hparams['BANDS']):, :, :] = new_conv1.weight[:, len(hparams['BANDS']):, :, :]
        new_conv1.requires_grad = True
        model.encoder.conv1 = new_conv1

    # set model in eval mode by default
    model.eval()
    
    return model



def convert_state_dict(ckpt, bands=None, torchvision_format=True, verbose=False):
    """ Accepts a checkpoint dictionary of DINO/MoCo-trained ResNet/ViT (or
    directly a state dict) and returns a state dict compatible with the
    torchvision/timm ResNet/ViT models. """

    if "teacher" in ckpt:
        ckpt = ckpt["teacher"]  # DINO
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]   # MoCo

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        # remove module. prefix
        if k.startswith("module."):
            if verbose: logger.info(f"Removing 'module' prefix: {k}")
            k = k.replace("module.", "")

        # MoCo v1/v2/v3
        if k.startswith("encoder_q."):
            if verbose: logger.info(f"Removing 'encoder_q' prefix: {k}")
            k = k.replace("encoder_q.", "")
        elif k.startswith("encoder_k."):
            if verbose: logger.info(f"Skipping layer with 'encoder_k' prefix: {k}")
            continue
        elif k.startswith("queue"):
            if verbose: logger.info(f"Skipping layer with 'queue' and 'queue_ptr' prefixes: {k}")
            continue
        
        # DINO: remove backbone. prefix (induced by DINO's MultiWrapper class in SSL4EO-S12 repo)
        if k.startswith("backbone."):
            if verbose: logger.info(f"Removing 'backbone' prefix: {k}")
            k = k.replace("backbone.", "")

        # DINO & MoCo: skip final head and fc layers
        if k.startswith("head") or k.startswith("fc"):
            if verbose: logger.info(f"Skipping fc head layer: {k}")
            continue
        
        # if specified, convert from timm to torchvision format
        if torchvision_format:
            if k.startswith("norm"):
                if verbose: logger.info(f"Converting norm layer: {k}")
                new_k = k.replace("norm.", "ln1.")
            elif k.startswith("patch_embed"):
                if verbose: logger.info(f"Converting patch_embed layer: {k}")
                if "proj" in k:
                    new_k = k.replace("proj", "projection")
                else:
                    new_k = k
            elif k.startswith("blocks"):
                if verbose: logger.info(f"Converting blocks layer: {k}")
                if "norm" in k:
                    new_k = k.replace("norm", "ln")
                elif "mlp.fc1" in k:
                    new_k = k.replace("mlp.fc1", "ffn.layers.0.0")
                elif "mlp.fc2" in k:
                    new_k = k.replace("mlp.fc2", "ffn.layers.1")
                elif "attn.qkv" in k:
                    new_k = k.replace("attn.qkv.", "attn.attn.in_proj_")
                elif "attn.proj" in k:
                    new_k = k.replace("attn.proj", "attn.attn.out_proj")
                else:
                    new_k = k
                new_k = new_k.replace("blocks.", "layers.")
            else:
                new_k = k
            new_ckpt[new_k] = v
        else:
            new_ckpt[k] = v

    # convert to specified L2A channels instead of 13 L1C channels
    # NB: if bands is None, we keep all 12 L2A channels
        
    s2_l1c_band_to_idx = {
        'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3,
        'B05': 4, 'B06': 5, 'B07': 6, 'B08': 7,
        'B8A': 8, 'B09': 9, 'B10': 10, 'B11': 11,
        'B12': 12,
    }

    if bands is None:
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]    # all S2-L2A bands
    else:
        indices = [s2_l1c_band_to_idx[b] for b in bands]
        
    if verbose: logger.info("Converting model to accept inputs of len(indices) channels")
    if "conv1.weight" in new_ckpt:  # resnet
        inputs = new_ckpt["conv1.weight"]
        if verbose: logger.info(f"Original conv1.weight shape: {inputs.shape}")
        new_ckpt["conv1.weight"] = inputs[:, indices, :, :]
        if verbose: logger.info(f"New conv1.weight shape: {new_ckpt['conv1.weight'].shape}")
    elif "patch_embed.projection.weight" in new_ckpt:   # vit
        inputs = new_ckpt["patch_embed.projection.weight"]
        if verbose: logger.info(f"Original patch_embed.projection.weight shape: {inputs.shape}")
        new_ckpt["patch_embed.projection.weight"] = inputs[:, indices, :, :]
        if verbose: logger.info(f"New patch_embed.projection.weight shape: {new_ckpt['patch_embed.projection.weight'].shape}")
    if verbose: logger.info("Conversion finished:")
    for k, v in new_ckpt.items():
        if verbose: logger.info(f"{k:<20s}: {v.shape}")
    return new_ckpt



def make_unet_bitemporal(model, fusion_module='diff_features', add_dem=None, **kwargs):
    """ Converts a single-temporal UNet model to a bitemporal one, by
    making the encoder siamese and adding the specified fusion module. """

    model.add_dem = add_dem

    # add fusion modules to the encoder object
    if fusion_module == 'ssma':
        eta = kwargs.get('eta', [10,25,42,73,128])
        assert len(eta) == model.encoder._depth, f'eta must have length {model.encoder._depth}'
        model.fusion_module = torch.nn.ModuleList([DummyLayer()] + [
            SSMA(
                channels = model.encoder._out_channels[i+1],
                eta = eta[i],
                ) for i in range(model.encoder._depth)
        ])

    elif fusion_module == 'concat_features':
        model.fusion_module = torch.nn.ModuleList([DummyLayer()] + [
            ConcatFeatures() for i in range(model.encoder._depth)
        ])

        # this fusion module changes the channel dimension of the feature maps,
        # so we need to change the decoder accordingly
        assert isinstance(model.decoder, UnetDecoder), f'only UnetDecoder is supported for now. Got {type(model.decoder)}'
        model.decoder = UnetDecoder(
            encoder_channels=tuple(2*c for c in model.encoder.out_channels),
            decoder_channels=(256, 128, 64, 32, 16),
            )
            
    elif fusion_module == 'diff_features':
        if model.add_dem == 'mid_concat':
            model.fusion_module = torch.nn.ModuleList([DummyLayer()] + [
                DiffFeatures(model.encoder.out_channels[i+1]) for i in range(model.encoder._depth)
            ])  # the argument is to inform DiffFeatures that it has to operate
        else:
            model.fusion_module = torch.nn.ModuleList([DummyLayer()] + [
                DiffFeatures() for _ in range(model.encoder._depth)
            ])
        # NB: in this case, the decoder is agnostic to the fusion module, since
        # it does not change the channel dimension of the feature maps
        if model.add_dem == 'late_concat':
            conv1 = torch.nn.Conv2d(
                in_channels = model.decoder.blocks[-1].conv2[0].weight.shape[0] + 2,    # ONLY UNET FOR NOW
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False,
            )
            bn1 = torch.nn.BatchNorm2d(64)
            relu1 = torch.nn.ReLU()
            conv2 = torch.nn.Conv2d(
                in_channels = 64,
                out_channels = model.decoder.blocks[-1].conv2[0].weight.shape[0],
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False,
            )
            bn2 = torch.nn.BatchNorm2d(model.decoder.blocks[-1].conv2[0].weight.shape[0])
            relu2 = torch.nn.ReLU()
            model.late_concat_conv = torch.nn.Sequential(conv1, bn1, relu1, conv2, bn2, relu2)

    elif fusion_module == 'absdiff_features':
        if model.add_dem == 'mid_concat':
            model.fusion_module = torch.nn.ModuleList([DummyLayer()] + [
                AbsDiffFeatures(model.encoder.out_channels[i+1]) for i in range(model.encoder._depth)
            ])  # the argument is to inform DiffFeatures that it has to operate
        else:
            model.fusion_module = torch.nn.ModuleList([DummyLayer()] + [
                AbsDiffFeatures() for _ in range(model.encoder._depth)
            ])
        # NB: in this case, the decoder is agnostic to the fusion module, since
        # it does not change the channel dimension of the feature maps
        if model.add_dem == 'late_concat':
            conv1 = torch.nn.Conv2d(
                in_channels = model.decoder.blocks[-1].conv2[0].weight.shape[0] + 2,    # ONLY UNET FOR NOW
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False,
            )
            bn1 = torch.nn.BatchNorm2d(64)
            relu1 = torch.nn.ReLU()
            conv2 = torch.nn.Conv2d(
                in_channels = 64,
                out_channels = model.decoder.blocks[-1].conv2[0].weight.shape[0],
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False,
            )
            bn2 = torch.nn.BatchNorm2d(model.decoder.blocks[-1].conv2[0].weight.shape[0])
            relu2 = torch.nn.ReLU()
            model.late_concat_conv = torch.nn.Sequential(conv1, bn1, relu1, conv2, bn2, relu2)
    
    elif fusion_module == 'changeformer_diff_features':
        model.fusion_module = torch.nn.ModuleList([DummyLayer()] + [
            ChangeformerDiffFeatures(
                in_channels=2*model.encoder._out_channels[i+1],
                out_channels=model.encoder._out_channels[i+1]
                ) for i in range(model.encoder._depth)
        ])

    else:
        raise NotImplementedError(f'fusion module {fusion_module} not implemented')



    def forward_bitemporal(x1, x2, dem):
        model.check_input_shape(x1)
        model.check_input_shape(x2)
        model.check_input_shape(dem)

        if model.add_dem == 'early_concat':
            x1 = torch.cat([x1, dem], dim=1)
            x2 = torch.cat([x2, dem], dim=1)

        # extract features
        features1 = model.encoder(x1)
        features2 = model.encoder(x2)

        # apply fusion modules
        if model.add_dem == 'mid_concat':
            features = [model.fusion_module[i](features1[i], features2[i], dem) for i in range(model.encoder._depth+1)]
        else:
            features = [model.fusion_module[i](features1[i], features2[i]) for i in range(model.encoder._depth+1)]

        # decode
        decoder_output = model.decoder(*features)

        if add_dem == 'late_concat':
            decoder_output = torch.cat([decoder_output, dem], dim=1)
            decoder_output = model.late_concat_conv(decoder_output)

        # segment
        masks = model.segmentation_head(decoder_output)

        # classify
        if model.classification_head is not None:
            labels = model.classification_head(features[-1])
            return masks, labels

        return masks
    
    @torch.no_grad()
    def predict_bitemporal(x1, x2, dem):
        if model.training:
            model.eval()

        x = model.forward(x1, x2, dem)

        return x
    
    # replace model's single-temporal forward and predict methods with
    # bitemporal forward and predict
    model.forward = forward_bitemporal
    model.predict = predict_bitemporal

    return model
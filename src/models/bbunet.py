import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead



class FusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        bn2 = torch.nn.BatchNorm2d(out_channels)
        self.fusion = torch.nn.Sequential(conv1, bn1, self.relu, conv2, bn2) # (batch, out_channels, size, size)

    def forward(self, x1, x2, dem):
        # compute difference between S2 feature maps
        diffs = x2 - x1

        # concatenate the diffs and the DEM feature maps
        f = torch.cat([diffs, dem], dim=1)

        # apply the fusion module
        f = self.fusion(f)

        # apply skip connection (fig. 2 of https://arxiv.org/pdf/1512.03385)
        #f = self.relu(f + diffs)
        f = f + diffs

        return f

class DummyLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x1, x2, dem):
        return torch.empty_like(x1)

def initialize_fusion_module(module_list):
    for module in module_list:
        if isinstance(module, FusionModule):
            fm = module.fusion
            for m in fm:
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
    return module_list

class BBUnet(SegmentationModel):
    def __init__(self, s2_in_channels=12, dem_in_channels=4, classes=1, *args, **kwargs):
        # currently, only resnet18+resnet18 is supported (as in the paper)
        super().__init__()

        # initialize S2 encoder
        self.encoder = get_encoder(
            name = "resnet18",
            encoder_weights = None,         # we will load the pretrained weights later
            in_channels = s2_in_channels,   # s2 encoder input channels (12 for Sentinel-2 level 2A)
            depth=5,
        )

        # initialize DEM encoder
        self.dem_encoder = get_encoder(
            name = "resnet18",
            encoder_weights = None,         # we will load the pretrained weights later
            in_channels = dem_in_channels,  # dem encoder input channels (4 for DEM)
            depth=5,
        )

        # add fusion module
        s2_f_channels = self.encoder.out_channels         # (12, 64, 256, 512, 1024, 2048) in the case of ResNet50, (12, 64, 64, 128, 256, 512) in the case of ResNet18
        dem_f_channels = self.dem_encoder.out_channels    # (4, 64, 64, 128, 256, 512) in the case of ResNet18

        in_channels = tuple(i+j for i,j in zip(s2_f_channels, dem_f_channels))
        self.fusion_module = torch.nn.ModuleList([
            DummyLayer(),
            FusionModule(in_channels=in_channels[1], out_channels=s2_f_channels[1]),
            FusionModule(in_channels=in_channels[2], out_channels=s2_f_channels[2]),
            FusionModule(in_channels=in_channels[3], out_channels=s2_f_channels[3]),
            FusionModule(in_channels=in_channels[4], out_channels=s2_f_channels[4]),
            FusionModule(in_channels=in_channels[5], out_channels=s2_f_channels[5]),
            ])

        # initialize decoder
        decoder_channels = (256, 128, 64, 32, 16)
        self.decoder = UnetDecoder(
            encoder_channels = self.encoder.out_channels,
            decoder_channels = decoder_channels,
            n_blocks = 5,
            use_batchnorm = True,
            center = False,
            attention_type = 'scse',
        )

        # initialize segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels = decoder_channels[-1],
            out_channels = classes,
            activation = None,
            kernel_size = 3,
        )

        # initialize classification head
        self.classification_head = None
        
        # initialize weights
        self.initialize()
        initialize_fusion_module(self.fusion_module)

        self.name = f"bbunet-r18"
        
    # make S2 encoder bitemporal and bimodal
    def forward(self, x1, x2, dem):
        self.check_input_shape(x1)
        self.check_input_shape(x2)
        self.check_input_shape(dem)

        # extract features
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)
        features_dem = self.dem_encoder(dem)

        # apply fusion modules
        features = [self.fusion_module[i](features1[i], features2[i], features_dem[i]) for i in range(self.encoder._depth+1)]

        # decode
        decoder_output = self.decoder(*features)

        # segment
        masks = self.segmentation_head(decoder_output)

        # classify
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
    
    @torch.no_grad()
    def predict(self, x1, x2, dem):
        if self.training:
            self.eval()

        x = self.forward(x1, x2, dem)

        return x
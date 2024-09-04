import torch
from torch.utils.model_zoo import load_url
import segmentation_models_pytorch as smp
import xarray as xr


class CloudDetector(torch.nn.Module):
    """
    SENCLOUD12 cloud detector for Sentinel-2.
    Under the hood, it is a UNet with MobileNetV2 encoder.
    Model from: https://github.com/earthnet2021/earthnet-minicuber/blob/main/earthnet_minicuber/provider/s2/cloudmask.py
    """

    def __init__(self, cloud_mask_rescale_factor=None, device=torch.device("cpu")) -> None:
        super().__init__()

        # the bands used by the cloud detector
        self.all_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "AOT", "WVP", "SCL"]
        self.bands = [b for b in self.all_bands if b not in {"B08", "SCL"}]

        # bands scale (will be used for data normalization)
        bands_scale = 12*[10000,]+[65535,65535,1]
        self.bands_scale = xr.DataArray(bands_scale, coords={'band':self.all_bands})
        #self.bands_scale = xr.Dataset(dict(zip(bands, bands_scale)))    # to xarray.Dataset, see ALTERNATIVE METHOD FOR PREPROCESSING

        # cloud mask rescale factor
        self.cloud_mask_rescale_factor = cloud_mask_rescale_factor
        
        # load cloud detector architecture
        self.model = smp.Unet(
            encoder_name = "mobilenet_v2",
            encoder_weights = None,         # we'll load our own weights later
            classes = 4,                    # 4 classes: 
            in_channels = len(self.bands)   # input dimension = n. of bands
        )

        # load cloud detector pretrained weights
        ckpt = load_url("https://nextcloud.bgc-jena.mpg.de/s/qHKcyZpzHtXnzL2/download/mobilenetv2_l2a_all.pth")
        self.model.load_state_dict(ckpt)

        # set cloud detector in eval mode and bring to device
        self.device = device
        self.model.eval()
        self.model.to(self.device)


    def forward(self, stack: xr.DataArray):
        """
            stack: xr.DataArray (13+, H, W)

        Returns:
            uint8 np.array (H, W) with the following interpretation:
                0: clear sky
                1: thick cloud
                2: thin cloud
                3: cloud shadow
        """
        assert set(self.bands).issubset(list(stack.long_name)), f"The given bands {list(stack.long_name)} do not contain the necessary bands for cloud masking. Please include bands {self.bands}."

        # rename bands with their long names (B01, B02, ...)
        stack = stack.assign_coords({'band' : list(stack.long_name)})

        # select 13 out of the 15 bands and rearrange their order according to self.bands
        stack = stack.sel(band=self.bands)

        # scale data
        stack = (stack / self.bands_scale).astype("float32")
        # NB: binary operations between DataArrays are performed
        # "coordinate-wise", so it would work even if we didn't select and
        # rearrange bands in the wanted order

        """ ALTERNATIVE APPROACH TO PREPROCESSING WITH xarray.Dataset

        # from pystac.item.Item to xarray.DataArray
        #stack = stack.to_dataset("band")

        # rename variables (i.e. bands) with their long names (B01, B02, ...)
        #old_names = list(stack.data_vars)
        #new_names = stack.long_name
        #stack = stack.rename_vars(dict(zip(old_names, new_names)))

        # change band order
        #stack = stack[self.bands]

        # scale data
        stack = (stack / self.bands_scale).astype("float32")

        # back to xarray.DataArray
        # stack = stack.to_array(dim='band)
        """

        # to torch.Tensor
        stack = torch.from_numpy(stack.values).to(self.device)

        # get shape (13, h, w)
        _, h, w = stack.shape

        # prepend dimension --> (1, 13, h, w)
        stack = stack.unsqueeze(0)

        # pad stack to be divisible by 32
        h_pad = ((h//32 + 1)*32)
        h_pad_left = (h_pad - h)//2
        h_pad_right = ((h_pad - h) + 1)//2
        w_pad = ((w//32 + 1)*32)
        w_pad_left = (w_pad - w)//2
        w_pad_right = ((w_pad - w) + 1)//2

        stack = torch.nn.functional.pad(
            stack,
            (w_pad_left, w_pad_right, h_pad_left, h_pad_right),
            mode = "reflect"
            )

        # if cloud_mask_rescale_factor is not None, expand stack (superresolution)
        if self.cloud_mask_rescale_factor is not None:
            stack = torch.nn.functional.interpolate(
                stack,
                scale_factor = self.cloud_mask_rescale_factor,
                mode = 'bilinear'
                )

        # detect clouds --> (1, 4, h, w)
        # (one confidence map for each of the 4 categories)
        with torch.no_grad():
            y_hat = self.model(stack)

        # get most confident label for each pixel --> (1, h, w)
        y_hat = torch.argmax(y_hat, dim = 1).float()

        # if cloud_mask_rescale_factor is not None, shrink mask (max-pooling downsampling)
        if self.cloud_mask_rescale_factor is not None:
            y_hat = torch.nn.functional.max_pool2d(
                y_hat[:,None,...],
                kernel_size = self.cloud_mask_rescale_factor
                )[:,0,...]
            #torch.nn.functional.interpolate(y_hat, size = (h,w), mode = "bilinear")

        # remove padding          
        y_hat = y_hat[:, h_pad_left:-h_pad_right, w_pad_left:-w_pad_right]

        return y_hat.cpu().numpy().astype("uint8")

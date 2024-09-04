import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xarray as xr



def visualize(
        ds,
        i,
        model,
        threshold=0.5,
        bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
        pre_divide=3000,
        post_divide=3000,
        compare_pred_gt=False,
        true_pos_color=[255,0,0],
        false_neg_color=[0,255,0],
        false_pos_color=[0,0,255],
        cloud_color=[255,255,255],
        post_alpha=1,
        mask_alpha=0.5,
        ):
    # load sample
    sample = ds[i]
    pre = sample['pre']
    pre_cm = ds.read(i)['pre_cm']
    #pre = torch.zeros_like(sample['pre'])
    post = sample['post']
    post_cm = ds.read(i)['post_cm']
    #post = torch.zeros_like(sample['post'])
    mask = sample['mask']
    dem = sample['dem']
    slope = sample['slope']
    aspect = sample['aspect']
    dem = torch.cat([dem, slope, aspect], dim=0)

    # inference
    logits = model(
        pre.unsqueeze(0).to(0 if torch.cuda.is_available() else 'cpu'),
        post.unsqueeze(0).to(0 if torch.cuda.is_available() else 'cpu'),
        dem.unsqueeze(0).to(0 if torch.cuda.is_available() else 'cpu'),
        )   # get logits
    out = (torch.sigmoid(logits) > threshold).squeeze(0).cpu().detach()   # apply sigmoid and threshold at 0.5
    #out = torch.where(mask != -1, out, -1)  # add cloud cover annotations

    # torch CHW -> numpy HWC
    pre_cm = pre_cm.transpose(1,2,0)
    post_cm = post_cm.transpose(1,2,0)
    out = out.numpy()[[-1]].transpose(1,2,0)
    mask = mask.numpy().transpose(1,2,0)
    dem = dem.numpy()[[0]].transpose(1,2,0)
    slope = slope.numpy().transpose(1,2,0)
    aspect = aspect.numpy().transpose(1,2,0)

    # all bands -> numpy HWC RGB
    B = np.where(np.array(bands) == 'B02')[0].item()
    G = np.where(np.array(bands) == 'B03')[0].item()
    R = np.where(np.array(bands) == 'B04')[0].item()
    pre_img = ds.read(i)['pre'][[R,G,B]]
    post_img = ds.read(i)['post'][[R,G,B]]
    poly = ds.read(i)["poly"]
    #r_pre, g_pre, b_pre = xr.DataArray(pre_img[0], dims=['y','x']), xr.DataArray(pre_img[1], dims=['y','x']), xr.DataArray(pre_img[2], dims=['y','x'])
    #r_post, g_post, b_post = xr.DataArray(post_img[0], dims=['y','x']), xr.DataArray(post_img[1], dims=['y','x']), xr.DataArray(post_img[2], dims=['y','x'])
    pre_img = pre_img.transpose(1,2,0) / pre_divide
    post_img = post_img.transpose(1,2,0) / post_divide
    poly = poly.transpose(1,2,0) / 255



    # plot pre
    plt.figure(figsize=(30,30))
    plt.imshow(pre_img)
    plt.title('pre')
    #plt.axis("off")
    plt.show()

    # plot post
    plt.figure(figsize=(30,30))
    plt.imshow(post_img)
    plt.title('post')
    #plt.axis("off")
    plt.show()

    # overlay gt mask (without clouds) on post
    _mask = np.zeros(poly.squeeze().shape + (4,))
    # add landslides
    _mask[:,:,[0,1,2,3]] = np.where(poly == 1, [*true_pos_color,255], _mask[:,:,[0,1,2,3]])
    _mask = Image.fromarray(_mask.astype(np.uint8))
    plt.figure(figsize=(30,30))
    plt.imshow(post_img, alpha=post_alpha)
    plt.imshow(_mask, alpha=mask_alpha)
    plt.title('ground truth')
    #plt.axis("off")
    plt.show()

    # overlay predicted mask on post
    _out = np.zeros(out.squeeze().shape + (4,))
    if compare_pred_gt:
        # add true positives
        _out[:,:,[0,1,2,3]] = np.where((out == 1) & (mask == 1), [*true_pos_color,255], _out[:,:,[0,1,2,3]])
        # add false negatives
        _out[:,:,[0,1,2,3]] = np.where((out == 0) & (mask == 1), [*false_neg_color,255], _out[:,:,[0,1,2,3]])
        # add false positives
        _out[:,:,[0,1,2,3]] = np.where((out == 1) & (mask == 0), [*false_pos_color,255], _out[:,:,[0,1,2,3]])
    else:
        # add landslides
        _out[:,:,[0,1,2,3]] = np.where(out == 1, [*true_pos_color,255], _out[:,:,[0,1,2,3]])
    # add thick clouds
    _out[:,:,[0,1,2,3]] = np.where((pre_cm == 1) | (post_cm == 1), [*cloud_color,255], _out[:,:,[0,1,2,3]])
    _out = Image.fromarray(_out.astype(np.uint8))
    plt.figure(figsize=(30,30))
    plt.imshow(post_img, alpha=post_alpha)
    plt.imshow(_out, alpha=mask_alpha)
    plt.title('predicted')
    #plt.axis("off")
    plt.show()



def get_images_for_tb(ds, i, model=None, threshold=0.5, bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']):
    # load sample
    sample = ds[i]
    pre = sample['pre']
    pre_cm = ds.read(i)['pre_cm']
    #pre = torch.zeros_like(sample['pre'])
    post = sample['post']
    post_cm = ds.read(i)['post_cm']
    #post = torch.zeros_like(sample['post'])
    mask = sample['mask']
    dem = sample['dem']
    slope = sample['slope']
    aspect = sample['aspect']
    dem = torch.cat([dem, slope, aspect], dim=0)

    # inference
    if model is not None:
        logits = model(
            pre.unsqueeze(0).to(0 if torch.cuda.is_available() else 'cpu'),
            post.unsqueeze(0).to(0 if torch.cuda.is_available() else 'cpu'),
            dem.unsqueeze(0).to(0 if torch.cuda.is_available() else 'cpu'),
            )   # get logits
        out = (torch.sigmoid(logits) > threshold).squeeze(0).cpu().detach()   # apply sigmoid and threshold at 0.5
        #out = torch.where(mask != -1, out, -1)  # add cloud cover annotations

    # torch CHW -> numpy HWC
    pre_cm = pre_cm.transpose(1,2,0)
    post_cm = post_cm.transpose(1,2,0)
    if model is not None:
        out = out.numpy()[[-1]].transpose(1,2,0)
    mask = mask.numpy().transpose(1,2,0)
    dem = dem.numpy()[[0]].transpose(1,2,0)
    slope = slope.numpy().transpose(1,2,0)
    aspect = aspect.numpy().transpose(1,2,0)

    # all bands -> numpy HWC RGB
    B = np.where(np.array(bands) == 'B02')[0].item()
    G = np.where(np.array(bands) == 'B03')[0].item()
    R = np.where(np.array(bands) == 'B04')[0].item()
    pre_img = ds.read(i)['pre'][[R,G,B]]
    post_img = ds.read(i)['post'][[R,G,B]]
    #r_pre, g_pre, b_pre = xr.DataArray(pre_img[0], dims=['y','x']), xr.DataArray(pre_img[1], dims=['y','x']), xr.DataArray(pre_img[2], dims=['y','x'])
    #r_post, g_post, b_post = xr.DataArray(post_img[0], dims=['y','x']), xr.DataArray(post_img[1], dims=['y','x']), xr.DataArray(post_img[2], dims=['y','x'])
    pre_img = pre_img.transpose(1,2,0) / 3000
    post_img = post_img.transpose(1,2,0) / 3000
    # for viz purposes, and to avoid the "clip" warning, clip explicitly to [0,1]
    pre_img = np.clip(pre_img, 0, 1)
    post_img = np.clip(post_img, 0, 1)

    # get pre image for tensorboard
    plt.imshow(pre_img)
    plt.axis("off")
    plt.tight_layout()
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    pre_img_plt = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    plt.cla()

    # get post image for tensorboard
    plt.imshow(post_img)
    plt.axis("off")
    plt.tight_layout()
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    post_img_plt = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    plt.cla()

    # get ground truth mask for tensorboard
    _mask = np.zeros(mask.squeeze().shape + (4,))
    # add landslides
    _mask[:,:,[0,1,2,3]] = np.where(mask == 1, [255,0,0,255], _mask[:,:,[0,1,2,3]])
    # add thick clouds
    _mask[:,:,[0,1,2,3]] = np.where(pre_cm == 1, [255,255,255,255], _mask[:,:,[0,1,2,3]])
    _mask[:,:,[0,1,2,3]] = np.where(post_cm == 1, [255,255,255,255], _mask[:,:,[0,1,2,3]])
    _mask = Image.fromarray(_mask.astype(np.uint8))
    plt.imshow(post_img)
    plt.imshow(_mask, alpha=0.5)
    plt.axis("off")
    plt.tight_layout()
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    gtmask_plt = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    plt.cla()

    # get predicted mask for tensorboard
    if model is not None:
        _out = np.zeros(out.squeeze().shape + (4,))
        # add landslides
        _out[:,:,[0,1,2,3]] = np.where(out == 1, [255,0,0,255], _out[:,:,[0,1,2,3]])
        # add thick clouds
        _out[:,:,[0,1,2,3]] = np.where(pre_cm == 1, [255,255,255,255], _out[:,:,[0,1,2,3]])
        _out[:,:,[0,1,2,3]] = np.where(post_cm == 1, [255,255,255,255], _out[:,:,[0,1,2,3]])
        _out = Image.fromarray(_out.astype(np.uint8))
        plt.imshow(post_img)
        plt.imshow(_out, alpha=0.5)
        plt.axis("off")
        plt.tight_layout()
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        predmask_plt = data.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.clf()
        plt.cla()
        plt.close()
    
    tb_imgs = {
        "pre": pre_img_plt,
        "post": post_img_plt,
        "gt": gtmask_plt,
    }

    if model is not None:
        tb_imgs["pred"] = predmask_plt

    return tb_imgs

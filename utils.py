import torch,rawpy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def apply_wb(org_img,pred,pred_type):
    """
    By using pred tensor (illumination map or uv),
    apply wb into original image (3-channel RGB image).
    """

    pred_rgb = torch.zeros_like(org_img) # b,c,h,w
    epsilon = 1e-3
    
    if org_img.ndim == 4:
        if pred_type == "illumination":
            pred_rgb[:,1,:,:] = org_img[:,1,:,:]
            pred_rgb[:,0,:,:] = org_img[:,0,:,:] / (pred[:,0,:,:] + epsilon)    # R_wb = R / illum_R
            pred_rgb[:,2,:,:] = org_img[:,2,:,:] / (pred[:,2,:,:] + epsilon)    # B_wb = B / illum_B
        elif pred_type == "uv":
            pred_rgb[:,1,:,:] = org_img[:,1,:,:]
            pred_rgb[:,0,:,:] = torch.exp(pred[:,0,:,:] + torch.log(org_img[:,1,:,:] + epsilon)) - epsilon
            pred_rgb[:,2,:,:] = torch.exp(pred[:,1,:,:] + torch.log(org_img[:,1,:,:] + epsilon)) - epsilon
    
    elif org_img.ndim == 3:
        if pred_type == "illumination":
            pred_rgb[1,:,:] = org_img[1,:,:]
            pred_rgb[0,:,:] = org_img[0,:,:] / (pred[0,:,:] + epsilon)    # R_wb = R / illum_R
            pred_rgb[2,:,:] = org_img[2,:,:] / (pred[2,:,:] + epsilon)    # B_wb = B / illum_B
        elif pred_type == "uv":
            pred_rgb[1,:,:] = org_img[1,:,:]
            pred_rgb[0,:,:] = torch.exp(pred[0,:,:] + torch.log(org_img[1,:,:] + epsilon)) - epsilon
            pred_rgb[2,:,:] = torch.exp(pred[1,:,:] + torch.log(org_img[1,:,:] + epsilon)) - epsilon
    
    return pred_rgb

def uv2rgb(x):
    return torch.cat([torch.exp(x[:,:1,:,:]),
                      torch.ones_like(x[:,:1,:,:]),
                      torch.exp(x[:,1:,:,:])],dim=1)

def rgb2uvl(img_rgb):
    """
    convert 3 channel rgb image into uvl
    """
    epsilon = 1e-3
    img_uvl = np.zeros_like(img_rgb, dtype='float32')
    img_uvl[:,:,2] = np.log(img_rgb[:,:,1] + epsilon)
    img_uvl[:,:,0] = np.log(img_rgb[:,:,0] + epsilon) - img_uvl[:,:,2]
    img_uvl[:,:,1] = np.log(img_rgb[:,:,2] + epsilon) - img_uvl[:,:,2]

    return img_uvl

def plot_illum(pred_map=None,gt_map=None,MAE_illum=None,MAE_rgb=None,PSNR=None):
    """
    plot illumination map into R,B 2-D space
    """
    fig = plt.figure()
    if pred_map is not None:
        plt.plot(pred_map[:,0],pred_map[:,1],'ro',alpha=0.03,markersize=5)
    if gt_map is not None:
        plt.plot(gt_map[:,0],gt_map[:,1],'bo',alpha=0.01,markersize=3)
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.title(f'MAE_illum:{MAE_illum:.4f} / PSNR:{PSNR}')
    plt.close()

    fig.canvas.draw()

    return np.array(fig.canvas.renderer._renderer)

def vis_illum(illum_map_rgb):
    """
    Visualize illumination map
    """
    return

def draw_AE_map(ae_map):
    fig = plt.figure()

    plt.pcolor(ae_map, vmin=0, vmax=60)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(f"MAE: {ae_map.mean():.5f}")
    plt.close()

    fig.canvas.draw()

    return np.array(fig.canvas.renderer._renderer)

def mix_chroma(mixmap,chroma_list,illum_count):
    """
    Mix illuminant chroma according to mixture map coefficient
    mixmap      : (w,h,c) - c is the number of valid illuminant
    chroma_list : (3 (RGB), 3 (Illum_idx))
                  contains R,G,B value or 0,0,0
    illum_count : contains valid illuminant number (1,2,3)
    """
    ret = np.stack((np.zeros_like(mixmap[:,:,0]),)*3, axis=2)
    for i in range(len(illum_count)):
        illum_idx = int(illum_count[i])-1
        mixmap_3ch = np.stack((mixmap[:,:,i],)*3, axis=2)
        ret += (mixmap_3ch * [[chroma_list[illum_idx]]])
    
    return ret

def cameraraw2srgb(patch,templete,wb_mode,no_auto_bright=False,ispred=False):
    """
    Visualize model inference result.
    1. Re-bayerize RGB image by duplicating G pixels.
    2. Copy bayer pattern image into rawpy templete instance
    3. Use custom wb (daylight or maintain as it is) to render RGB image
    4. Crop proper size of patch from rendered RGB image
    """
    patch = patch.permute((1,2,0))          # c,w,h -> w,h,c
    height, width, _ = patch.shape
    
    raw = rawpy.imread(templete + ".dng")
    white_level = raw.white_level
    
    if templete == 'sony':
        black_level = 512
        white_level = raw.white_level / 4
    else:
        black_level = min(raw.black_level_per_channel)
        white_level = raw.white_level

    if ispred:
        rgb = np.clip(patch.cpu().numpy(), 0, white_level).astype('uint16')
    else:
        rgb = patch.cpu().numpy().astype('uint16')

    bayer = bayerize(rgb, templete, black_level)
    rendered = render(raw, white_level, bayer, height, width, wb_mode, no_auto_bright)

    return rendered

def visualize_grid(input_patch,pred_uvbranch,pred_ilbranch,pred_interpolate,
                   interpolation,score,gt,ae_maps,templete,no_auto_bright):
    """
    Generate 3 X 4 grid ouput visualization
    |   Input   |    GT     |    Pred (image or illumination map)   |    Error map   |
    | UV branch |   Score   |            Score * UV branch          | E map UVbranch |
    | IL branch |   Score   |            Score * IL branch          | E map ILbranch |
    """
    
    # process WB for each branch & interpolated one
    pred_rgb_uvbranch = apply_wb(input_patch,pred_uvbranch,pred_type='uv')
    pred_rgb_ilbranch = apply_wb(input_patch,pred_ilbranch,pred_type='uv')
    pred_rgb_interpolate = apply_wb(input_patch,pred_interpolate,pred_type='uv')
    
    # proceess WB for score applied image
    if interpolation == 'exp':
        pred_rgb_score_uvbranch = apply_wb(input_patch,pred_uvbranch*score,pred_type='uv').cpu()
        pred_rgb_score_ilbranch = apply_wb(input_patch,pred_ilbranch*(1-score),pred_type='uv').cpu()
    elif interpolation == 'linear':
        pred_rgb_score_uvbranch = (pred_rgb_uvbranch * score).cpu()
        pred_rgb_score_ilbranch = (pred_rgb_ilbranch * (1-score)).cpu()

    score_3ch = torch.cat([score,]*3,dim=0).permute((1,2,0)).cpu().numpy()

    # first row
    i_11 = cameraraw2srgb(patch=input_patch,templete=templete,wb_mode='daylight_wb')
    i_12 = cameraraw2srgb(patch=gt,templete=templete,wb_mode='maintain')
    i_13 = cameraraw2srgb(patch=pred_rgb_interpolate,templete=templete,wb_mode='maintain',ispred=True)
    i_14 = draw_AE_map(ae_maps[0].cpu())

    # second row
    i_21 = cameraraw2srgb(patch=pred_rgb_uvbranch,templete=templete,wb_mode='maintain',ispred=True)
    i_22 = score_3ch * 255.
    i_23 = cameraraw2srgb(patch=pred_rgb_score_uvbranch,templete=templete,wb_mode='maintain',ispred=True)
    i_24 = draw_AE_map(ae_maps[1].cpu())

    # third row
    i_31 = cameraraw2srgb(patch=pred_rgb_ilbranch,templete=templete,wb_mode='maintain',ispred=True)
    i_32 = (1-score_3ch) * 255.
    i_33 = cameraraw2srgb(patch=pred_rgb_score_ilbranch,templete=templete,wb_mode='maintain',ispred=True)
    i_34 = draw_AE_map(ae_maps[2].cpu())

    i_1 = np.hstack([i_11,i_12,i_13])
    i_2 = np.hstack([i_21,i_22,i_23])
    i_3 = np.hstack([i_31,i_32,i_33])
    i = np.vstack([i_1,i_2,i_3])

    ae_map_stack = np.vstack([i_14,i_24,i_34])

    return i, ae_map_stack


    pass

def bayerize(img_rgb, camera, black_level):
    h,w,c = img_rgb.shape

    bayer_pattern = np.zeros((h*2,w*2))
    
    if camera == "galaxy":
        bayer_pattern[0::2,1::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,2] # B
    elif camera == "sony" or camera == 'nikon':
        bayer_pattern[0::2,0::2] = img_rgb[:,:,0] # R
        bayer_pattern[0::2,1::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,0::2] = img_rgb[:,:,1] # G
        bayer_pattern[1::2,1::2] = img_rgb[:,:,2] # B

    return bayer_pattern + black_level

def render(raw, white_level, bayer, height, width, wb_method, no_auto_bright):
    raw_mat = raw.raw_image_visible
    for h in range(height*2):
        for w in range(width*2):
            raw_mat[h,w] = bayer[h,w]

    if wb_method == "maintain":
        user_wb = [1.,1.,1.,1.]
    elif wb_method == "daylight_wb":
        user_wb = raw.daylight_whitebalance

    rgb = raw.postprocess(user_sat=white_level, user_wb=user_wb, half_size=True, no_auto_bright=no_auto_bright)
    rgb_croped = rgb[0:height,0:width,:]
    
    return rgb_croped

def avg_pool(img, pool_size=1):
    return  F.avg_pool2d(input=img,kernel_size=pool_size, stride=pool_size)

def upsample(img, scale=1):
    return F.interpolate(input=img,scale_factor=scale,mode='bilinear',align_corners=True)
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import cv2

def plot_multihead_attention(att_maps, num_cols=4, save_path=None):
    """
    绘制多头注意力热力图
    :param att_maps: torch.Tensor or np.ndarray, shape (num_heads, H, W)
    :param num_cols: 每行显示多少个头
    :param save_path: 保存路径
    """
    if isinstance(att_maps, torch.Tensor):
        att_maps = att_maps.detach().cpu().numpy()

    num_heads = att_maps.shape[0]

    num_rows = (num_heads + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    axes = axes.flatten()

    multi_attns = []
    for idx in range(num_heads):
        att_map = att_maps[idx]
        att_map = (att_map - np.min(att_map)) / (np.max(att_map) - np.min(att_map) + 1e-8)
        multi_attns.append(att_map)

        axes[idx].imshow(att_map, cmap='jet')
        axes[idx].set_title(f'Head {idx+1}')
        axes[idx].axis('off')

    # Hide unused axes
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return multi_attns

    

def plot_attention_map(att_map, title='Attention Map', save_path=None):
    """
    绘制单张注意力热力图
    :param att_map: torch.Tensor or np.ndarray, shape (H, W)
    :param title: 标题
    :param save_path: 如果要保存图片，传保存路径
    """
    if isinstance(att_map, torch.Tensor):
        att_map = att_map.detach().cpu().numpy()

    # Normalize to [0,1]
    att_map = (att_map - np.min(att_map)) / (np.max(att_map) - np.min(att_map) + 1e-8)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(att_map, cmap='jet')
    plt.colorbar()
    # plt.title(title)
    plt.axis('off')

    # if save_path:
    #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    
    return att_map

def overlay_attention_on_image(img, att_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将单个注意力图叠加到原图上
    :param img: 原图，np.ndarray, shape (H, W, 3)，值在[0,255]
    :param att_map: 注意力图，np.ndarray, shape (H', W')，单通道
    :param alpha: 热力图透明度
    :param colormap: OpenCV colormap
    :return: 叠加后的图
    """
    H, W = img.shape[:2]
    
    # Normalize att_map
    att_map = (att_map - np.min(att_map)) / (np.max(att_map) - np.min(att_map) + 1e-8)

    # Resize to original image size
    att_map = cv2.resize(att_map, (W, H))

    # Apply colormap
    att_map_color = cv2.applyColorMap(np.uint8(255 * att_map), colormap)
    att_map_color = cv2.cvtColor(att_map_color, cv2.COLOR_BGR2RGB)  # OpenCV默认是BGR，要转RGB

    # Overlay
    overlay = cv2.addWeighted(img, 1 - alpha, att_map_color, alpha, 0)

    return overlay

def plot_multihead_attention_overlay(img, att_maps, num_cols=4, alpha=0.5, save_path=None):
    """
    绘制多头注意力叠加原图的效果
    :param img: 原图，np.ndarray, (H, W, 3)
    :param att_maps: 注意力图，torch.Tensor or np.ndarray, shape (num_heads, H', W')
    :param num_cols: 每行多少个子图
    :param alpha: 热力图透明度
    """
    if isinstance(att_maps, torch.Tensor):
        att_maps = att_maps.detach().cpu().numpy()

    num_heads = len(att_maps)
    num_rows = (num_heads + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    axes = axes.flatten()

    for idx in range(num_heads):
        overlay = overlay_attention_on_image(img, att_maps[idx], alpha=alpha)

        axes[idx].imshow(overlay)
        axes[idx].set_title(f'Head {idx+1}')
        axes[idx].axis('off')

    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    

       
   


image = cv2.imread('/root/autodl-tmp/visualization/TNBC/Attention_Map/01_3.png')
all_attn = torch.load('/root/autodl-tmp/visualization/TNBC/Attention_Map/01_3_attn.pt')
save_path = '/root/autodl-tmp/visualization/TNBC/Attention_Map/'
encoder_attn = all_attn['encoder'] # [16,2,1024] encoder_attn[0].shape
decoder_attn = all_attn['decoder'] # [8,5,1024]

num_layers = len(encoder_attn)
H=W=32

# 还是绘制多头注意力图比较好

# for i in range(num_layers):
#     q_attn = encoder_attn[i]
#     for j in range(q_attn.shape[1]):
#         q = q_attn[:,j,:]
#         q = q.reshape(-1, H, W)
#         save_path_q = save_path + f'encoder_attn_layer_{i}_{j}.png'
#         encoder_attns = plot_multihead_attention(q, num_cols=8, save_path=save_path_q)
#         plot_multihead_attention_overlay(image, encoder_attns, num_cols=8, alpha=0.6, save_path=save_path + f'encoder_attn_overlay_layer_{i}_{j}.png')
     

# for i in range(num_layers):
#     a_attn = decoder_attn[i].squeeze(0)
#     for j in range(a_attn.shape[1]):
#         a = a_attn[:,j,:]
#         a = a.reshape(-1,H, W)
#         save_path_a = save_path + f'decoder_attn_layer_{i}_{j}.png'
#         decoder_attns = plot_multihead_attention(a, num_cols=8, save_path=save_path_a)
#         plot_multihead_attention_overlay(image, decoder_attns, num_cols=8, alpha=0.6, save_path=save_path + f'decoder_attn_overlay_layer_{i}_{j}.png')

q = encoder_attn[3][:,1,:]
q = q.reshape(-1, H, W)[3]
q_map = plot_attention_map(q, title='Attention Map', save_path=save_path + 'encoder_attn_layer_3_1_3.png')
q_overlay = overlay_attention_on_image(image, q_map, alpha=0.5, colormap=cv2.COLORMAP_JET)

a = decoder_attn[0].squeeze(0)[:,4,:]
a = a.reshape(-1, H, W)[4]
a_map = plot_attention_map(a, title='Attention Map', save_path=save_path + 'decoder_attn_layer_0_4_4.png')
a_overlay = overlay_attention_on_image(image, a_map, alpha=0.5, colormap=cv2.COLORMAP_JET)

plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.imshow(image)
plt.subplot(3, 1, 2)
plt.imshow(q_overlay)
plt.subplot(3, 1, 3)
plt.imshow(a_overlay)
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.axis('off')

plt.tight_layout()
plt.savefig(save_path + 'overlay.png', bbox_inches='tight')
print("finish")

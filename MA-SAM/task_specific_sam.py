import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic
from typing import Type
import math

from typing import Any, Dict, List, Tuple

from segment_anything.modeling import Sam, TwoWayTransformer


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

class ImageEncoderViT_task(nn.Module):
    def __init__(
        self,
        ImageEncoderViT: nn.Module,
        init_layers,
        # task_adapter:nn.Module,
    ) -> None:
        super().__init__()
        self.ImageEncoderViT = ImageEncoderViT
        self.init_layers = init_layers
        self.img_size = self.ImageEncoderViT.img_size

    def forward(self, x: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        x = self.ImageEncoderViT.patch_embed(x)
        if self.ImageEncoderViT.pos_embed is not None:
            x = x + self.ImageEncoderViT.pos_embed

        outputs = []
        count = 0
        for i in range(len(self.ImageEncoderViT.blocks)):
            if i in self.init_layers:
                x = self.ImageEncoderViT.blocks[i](x, task_embed[count])
                count += 1
                outputs.append(x)
            else:
                x = self.ImageEncoderViT.blocks[i](x) 
                # if i == 0:
                #     outputs.append(x)
            

        x = self.ImageEncoderViT.neck(x.permute(0, 3, 1, 2)) #[B, C, H, W]
        

        return outputs

class Task_adapter(nn.Module):

    def __init__(
            self,
            num_mask_tokens:int,
            image_dim:int, 
            decoder_dim:int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        
        super().__init__()
        self.num_layers = num_layers
        self.num_mask_tokens = num_mask_tokens
        # h = [hidden_dim] * (num_layers - 1)
        self.task_adapter_mlp_list = nn.ModuleList()
        self.mask_adapter_mlp_list = nn.ModuleList()
        for i in range(self.num_layers):
            self.task_adapter_mlp_list.append(nn.Sequential(
                nn.Linear(decoder_dim, image_dim//4),
                nn.ReLU(),
                nn.Linear(image_dim//4, image_dim//4),
                nn.ReLU(),
                nn.Linear(image_dim//4, image_dim),
                nn.ReLU(),
                nn.Linear(image_dim, image_dim), #增加一项全连接层
                ) 
            )
        
    
    def forward(self, task_embed: torch.Tensor):
        image_task_embed = []
        mask_task_embed = []
        for i in range(self.num_layers):
            image_task_embed.append(self.task_adapter_mlp_list[i](task_embed[i])) # what if we do not give it mean[task_num, dim]
            mask_tokens = []
            for j in range(self.num_mask_tokens):
               mask_tokens.append(self.mask_adapter_mlp_list[i][j](task_embed[i][j]))
            mask_task_embed.append(torch.stack(mask_tokens))
        
        return image_task_embed, mask_task_embed

class Mask_adapter(nn.Module):
    def __init__(self, 
                num_mask_tokens:int,
                image_dim:int, 
                decoder_dim:int,
                num_layers: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_mask_tokens = num_mask_tokens
        
        
        self.mask_adapter_mlp_list = nn.ModuleList()
        self.neck_list = nn.ModuleList()
        for i in range(num_layers):
            neck = nn.Sequential(
                nn.Linear(image_dim, image_dim//4),
                nn.ReLU(),
                nn.Linear(image_dim//4, decoder_dim)
            )

            mask_adapter = nn.ModuleList(
                [
                    MLP(decoder_dim, decoder_dim//4, decoder_dim, 3)
                    for i in range(self.num_mask_tokens)
                ]
            )

            self.mask_adapter_mlp_list.append(mask_adapter)
            self.neck_list.append(neck)
    
    def forward(self, task_embed):
        mask_task_embed = []
        for i in range(self.num_layers):
            mask_tokens = []
            mask_embed = self.neck_list[i](task_embed[i])
            for j in range(self.num_mask_tokens):
               mask_tokens.append(self.mask_adapter_mlp_list[i][j](mask_embed[j]))
            mask_task_embed.append(torch.stack(mask_tokens))
        
        return mask_task_embed
        

class Block_task(nn.Module):
    def __init__(
            self,
            Block: nn.Module,
    ):
        super().__init__()
        self.Block = Block
    
    def forward(self, x:torch.Tensor, task_embed) -> torch.Tensor:
        shortcut = x
        x = self.Block.norm1(x)
        # Window partition
        if self.Block.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.Block.window_size)  # [B * num_windows, window_size, window_size, C]

        x = self.Block.attn(x, task_embed)
        # Reverse window partition
        if self.Block.window_size > 0:
            x = window_unpartition(x, self.Block.window_size, pad_hw, (H, W))

        x = shortcut + x

        x = x + self.Block.mlp(self.Block.norm2(x))

        return x

class Attention_task(nn.Module):
    def __init__(
            self,
            Attention: nn.Module,
    ):
        super().__init__()
        self.Attention = Attention
    
    def forward(self, x:torch.Tensor, task_embed:torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        task_num, dim = task_embed.shape
        # concate task_embed
        x = x.reshape(B, H*W, -1)
        task_embed = task_embed.expand(B, task_num, -1)
        x = torch.concat([x, task_embed], dim=-2) #[B, H*W+1, -1]
        # qkv with shape (3, B, nHead, H * W + 1, C)
        qkv = self.Attention.qkv(x).reshape(B, H * W + task_num, 3, self.Attention.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W + 1, C)
        q, k, v = qkv.reshape(3, B * self.Attention.num_heads, H * W + task_num, -1).unbind(0)

        attn = (q * self.Attention.scale) @ k.transpose(-2, -1) #[B * nHead, H*W+1, H*W+1] nheads=16

        if self.Attention.use_rel_pos:
            attn[:,:-task_num,:-task_num] = add_decomposed_rel_pos(attn[:, :-task_num, :-task_num], q[:, :-task_num, :], self.Attention.rel_pos_h, self.Attention.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        # x = (attn @ v).view(B, self.Attention.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = attn @ v
        x = x[:, :-task_num, :] #取消掉concate的东西
        x = x.view(B, self.Attention.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.Attention.proj(x)

        return x

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class _LoRA_qkv_global(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv

class MaskDecoder_task(nn.Module):
    def __init__(
            self,
            MaskDecoder: nn.Module,
            num_layer: int,
            transformer_dim: int,
            encoder_dim:int,
            mask_in_chans = 32,
    ):
        super().__init__()
        # self.MaskDecoder = MaskDecoder
        self.num_mask_tokens = MaskDecoder.num_mask_tokens
        self.num_layer = num_layer
        
        self.mask_transformer_list = nn.ModuleList()
        self.decoder_transformer_list = nn.ModuleList()
        self.mask_tokens_list = nn.ParameterList()
        self.output_upscaling_list = nn.ModuleList()
        self.output_hypernetworks_mlps_list = nn.ModuleList()
        self.mask_downscaling_list = nn.ModuleList()
      
        for i in range(num_layer-1):
            mask_transform = TwoWayTransformer(
                depth=1,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            )

            self.mask_transformer_list.append(mask_transform)

            mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
            self.mask_tokens_list.append(mask_tokens)

            output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                nn.GELU()
            )
            self.output_upscaling_list.append(output_upscaling)
            
            output_hypernetworks_mlps = nn.ModuleList(
                [
                    MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for i in range(self.num_mask_tokens)
                ]
            )

            self.output_hypernetworks_mlps_list.append(output_hypernetworks_mlps)

            mask_downscaling = nn.Sequential(
                nn.Softmax(dim=1),
                nn.Conv2d(self.num_mask_tokens, mask_in_chans // 4, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                nn.Conv2d(mask_in_chans, transformer_dim, kernel_size=1),
            )  # downsample to 1/4
            self.mask_downscaling_list.append(mask_downscaling)

        self.u_fusion = U_decoder(transformer_dim, num_layer, self.num_mask_tokens) # less than transformer module
           
    
    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            task_specific_embed: torch.Tensor,
    ):  
        
        masks_list = []
        for i in range(self.num_layer-1):
            masks = self.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                task_specific_embed = task_specific_embed, 
                index = i,
            )
            masks_list.append(masks)
        
        down_scale_masks = []
        for layer, masks in zip(self.mask_downscaling_list, masks_list):
            down_scale_masks.append(layer(masks)) 
        
        masks = self.u_fusion(image_embeddings,
                              image_pe, 
                              sparse_prompt_embeddings,
                              down_scale_masks)
                
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        task_specific_embed: torch.Tensor, # 加入可学习部分
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens (number of mask_tokens remains to 1)
        # output_tokens = torch.cat([self.iou_tokens.weight, self.mask_tokens.weight], dim=0)
        output_tokens = self.mask_tokens_list[index].weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) #[1, -1, -1]
        
        # Expand per-image data in batch direction to be per-mask
        mask_embed = task_specific_embed[index].unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        hs = torch.cat((output_tokens, sparse_prompt_embeddings, mask_embed), dim=1) 
        src = torch.repeat_interleave(image_embeddings[index], hs.shape[0], dim=0)
        src = src + dense_prompt_embeddings  
        b, c, h, w = src.shape
        src = src.flatten(2).permute(0,2,1)
        pos_src = torch.repeat_interleave(image_pe, hs.shape[0], dim=0)
    
        hs, src = self.mask_transformer_list[index](src, pos_src, hs)
        
        mask_tokens_out = hs[:, 0 : (0 + self.num_mask_tokens), :]
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling_list[index](src) #[b, ]
        # print(upscaled_embedding.shape)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps_list[index][i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, c, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
       
        return masks

    
class U_decoder(nn.Module):
    def __init__(
            self,
            transformer_dim:int,
            global_attn_num : int,
            num_mask_tokens:int,
    ):
        super().__init__()

        self.layer_num = global_attn_num
        self.decoder_transform_list = nn.ModuleList()
        self.mask_tokens = nn.Embedding(num_mask_tokens, transformer_dim)
        self.num_mask_tokens = num_mask_tokens

        for i in range(self.layer_num):
            decoder_transform = TwoWayTransformer(
                depth=1,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            )

            self.decoder_transform_list.append(decoder_transform)
       
        self.final_output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 8),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 16),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 16, transformer_dim // 32, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.final_output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        
    
    def forward(self, 
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                masks_list: list):
        
        for i in range(self.layer_num):
            output_tokens = self.mask_tokens.weight
            output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) #[1, -1, -1]
        
            # Expand per-image data in batch direction to be per-mask
            hs = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) 
            src = torch.repeat_interleave(image_embeddings[-1], hs.shape[0], dim=0)
            b, c, h, w = src.shape
            pos_src = torch.repeat_interleave(image_pe, hs.shape[0], dim=0)
            for i in range(self.layer_num-1, -1, -1):
                if i == self.layer_num-1:
                    # src = src + masks_list[i]
                    src = src.flatten(2).permute(0,2,1)
                else:
                    src = src + masks_list[i].flatten(2).permute(0, 2, 1)
                hs, src = self.decoder_transform_list[i](src, pos_src, hs)
            
            
            mask_tokens_out = hs[:, 0 : (0 + self.num_mask_tokens), :]
            # Upscale mask embeddings and predict masks using the mask tokens
            src = src.transpose(1, 2).view(b, c, h, w)
            upscaled_embedding = self.final_output_upscaling(src) #[b, ]
            # print(upscaled_embedding.shape)
            hyper_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(self.final_output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

            b, c, h, w = upscaled_embedding.shape  # [h, c, h, w]
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks
        
class Neck(nn.Module):
    def __init__(self,embed_dim, out_chans, global_attn_layer):
        super().__init__()
        # image_encoder_neck
        self.image_neck_list = nn.ModuleList()
        for i in range(global_attn_layer):
            self.image_neck_list.append(nn.Sequential(
                                            nn.Conv2d(
                                            embed_dim,
                                            out_chans,
                                            kernel_size=1,
                                            bias=False,
                                        ),
                                        LayerNorm2d(out_chans),
                                        nn.Conv2d(
                                            out_chans,
                                            out_chans,
                                            kernel_size=3,
                                            padding=1,
                                            bias=False,
                                        ),
                                        LayerNorm2d(out_chans),
                                        )
                                    )
        
    def forward(self, image_embeddings):
        for i in range(len(self.image_neck_list)):
            image_embeddings[i] = self.image_neck_list[i](image_embeddings[i].permute(0, 3, 1, 2))
        
        return image_embeddings


class Sam_task(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        sam_model: Sam,
        r: int,
        lora_layer = None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        # create task_specific embed
        
        
        decoder_dim = sam_model.mask_decoder.mask_tokens.weight.shape[1]
        image_encoder_dim = sam_model.image_encoder.pos_embed.shape[3]
        image_size = sam_model.image_encoder.pos_embed.shape[1] * 16 # vit_b: 32*16 = 512
        self.global_attn_num = len(sam_model.image_encoder.global_attn_indexes) # 4
        num_mask_tokens = sam_model.mask_decoder.num_mask_tokens
        
        self.task_adapter = Mask_adapter(num_mask_tokens, image_encoder_dim, decoder_dim, self.global_attn_num)
        self.Neck_list = Neck(image_encoder_dim, decoder_dim, self.global_attn_num)
        
        self.task_specific_embed_list = nn.ParameterList()

        # lora
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks))
            )
        
        self.w_As = []
        self.w_Bs = []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False


        for layer_i , blk in enumerate(sam_model.image_encoder.blocks):
            if layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            if layer_i in sam_model.image_encoder.global_attn_indexes:
                blk.attn.qkv = _LoRA_qkv_global(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
                blk.attn = Attention_task(blk.attn)
                sam_model.image_encoder.blocks[layer_i] = Block_task(blk)

                # task_specific_embed
                # task_specific_embed = torch.empty_like(sam_model.mask_decoder.mask_tokens.weight) #[task_num, decoder_embed]
                task_specific_embed = torch.empty(num_mask_tokens, image_encoder_dim) #[task_num, encoder_dim]
                nn.init.normal_(task_specific_embed, std=0.02)
                task_specific_embed = nn.Parameter(task_specific_embed)
                self.task_specific_embed_list.append(task_specific_embed)

            else:
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
                # sam_model.image_encoder[layer_i] = blk     
        
        sam_model.image_encoder = ImageEncoderViT_task(sam_model.image_encoder, sam_model.image_encoder.global_attn_indexes)
        self.mask_decoder = MaskDecoder_task(sam_model.mask_decoder, self.global_attn_num, decoder_dim, image_encoder_dim)
        
        self.sam = sam_model

        self.init_weights() 

    @property
    def device(self) -> Any:
        return self.sam.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size):
        
        outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size):
        b, h, w = batched_input.shape[0], batched_input.shape[2], batched_input.shape[3] # [b, 3, h, w]
        batched_input = batched_input.contiguous().view(-1, 3, h, w) #[b, 3, h, w]

        input_images = self.sam.preprocess(batched_input)

        # get image and mask task_embeds
        mask_task_embed = self.task_adapter(self.task_specific_embed_list)
        
        image_embeddings = self.sam.image_encoder(input_images, self.task_specific_embed_list) #
        image_embeddings = self.Neck_list(image_embeddings) #[image_embed_dim -> decoder_embed_dim]
        
        # prompt encoder
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None,
        ) #[batch, 256, 32, 32]

        low_res_masks = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            task_specific_embed = mask_task_embed,
        )

        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': None,
            'low_res_logits': low_res_masks
        }
    
        return outputs
    
    def init_weights(self):
        # task_adapter = self.task_adapter.neck_list
        # mask_adapter = self.task_adapter.mask_adapter_mlp_list
        # layers = len(task_adapter)
        # for layer in range(layers):
        #     nn.init.constant_(task_adapter[layer][-1].weight, 0)
        #     nn.init.constant_(task_adapter[layer][-1].bias, 0)
            
        #  #init the mask_adapter
        #     for item in mask_adapter[layer]:
        #         nn.init.constant_(item.layers[-1].weight, 0)
        #         nn.init.constant_(item.layers[-1].weight, 0)
        
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
        

    def save_parameters(self, filename: str) ->None:
        
        assert filename.endswith(".pt") or filename.endswith('.pth')
        num_task = self.global_attn_num
        task_embed_tensors = {f"task_specific_embed_{i:03d}": self.task_specific_embed_list[i] for i in range(num_task)}

        # lora
        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        task_adapter_tensors = {}
        neck_list_tensors = {}
        prompt_encoder_tensors = {}
        # u_decoder_tensors = {}
        mask_decoder_tensors = {}
        # mask_adapter_tensors = {}

        
        if isinstance(self, torch.nn.DataParallel) or isinstance(self, torch.nn.parallel.DistributedDataParallel):
            self_state_dict = self.module.state_dict()
        else:
            self_state_dict = self.state_dict()

        for key, value in self_state_dict.items():
            if 'Neck_list' in key:
                neck_list_tensors[key] = value
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'task_adapter' in key:
                task_adapter_tensors[key] = value
            if 'mask_decoder' in key and 'sam' not in key:
                mask_decoder_tensors[key] = value
        

        merged_dict = {**a_tensors, **b_tensors,**task_embed_tensors, **task_adapter_tensors,  **neck_list_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)
    
    def load_parameters(self, filename: str) -> None:
        
        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = nn.Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = nn.Parameter(saved_tensor)

        sam_dict = self.state_dict() #调整为针对self的字典
        sam_keys = sam_dict.keys()

        # load task_specific_embed
        for i, task_embed in enumerate(self.task_specific_embed_list):
            saved_key = f"task_specific_embed_{i:03d}"
            saved_tensor = state_dict[saved_key]
            task_embed = nn.Parameter(saved_tensor)
        
        # load task_adapter
        task_adapter_keys = [k for k in sam_keys if 'task_adapter' in k]
        task_adapter_values = [state_dict[k] for k in task_adapter_keys]
        task_adapter_state_dict = {k:v for k, v in zip(task_adapter_keys, task_adapter_values)}
        sam_dict.update(task_adapter_state_dict)

        # load neck_list
        neck_list_keys = [k for k in sam_keys if 'Neck_list' in k]
        neck_list_values = [state_dict[k] for k in neck_list_keys]
        neck_list_state_dict = {k:v for k, v in zip(neck_list_keys, neck_list_values)}
        sam_dict.update(neck_list_state_dict)

        # load prompt_encoder
        prompt_encoder_keys = [k for k in sam_keys if 'u_decoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_state_dict = {k:v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_state_dict)

        # load mask_decoder
        mask_decoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_state_dict = {k:v for k,v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_state_dict)


        self.load_state_dict(sam_dict)



    

    
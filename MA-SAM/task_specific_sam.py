import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from segment_anything.modeling import Sam, LayerNorm2d, MLPBlock



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
        # task_adapter:nn.Module,
    ) -> None:
        super().__init__()
        self.ImageEncoderViT = ImageEncoderViT
        self.img_size = self.ImageEncoderViT.img_size
        # self.task_adapter = self.task_adapter = Task_adapter(out_chans, embed_dim//4, embed_dim, len(global_attn_indexes))
        # self.task_adapter = task_adapter

    def forward(self, x: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        x = self.ImageEncoderViT.patch_embed(x)
        # task_adapter_embeddings = self.task_adapter(task_embed)  #[layers, task_num, dim]
        if self.ImageEncoderViT.pos_embed is not None:
            x = x + self.ImageEncoderViT.pos_embed

        # for blk in self.blocks:
        #     x = blk(x)
        outputs = []
        count = 0
        for i in range(len(self.ImageEncoderViT.blocks)):
            if i in self.ImageEncoderViT.global_attn_indexes:
                x = self.ImageEncoderViT.blocks[i](x, task_embed[count])
                count += 1
                outputs.append(x)
            else:
                x = self.ImageEncoderViT.blocks[i](x) 
            

        x = self.ImageEncoderViT.neck(x.permute(0, 3, 1, 2)) #[B, C, H, W]

        return x

class Task_adapter(nn.Module):

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
        # h = [hidden_dim] * (num_layers - 1)
        self.task_adapter_mlp_list = nn.ModuleList()
        for i in range(self.num_layers):
            self.task_adapter_mlp_list.append(nn.Sequential(
                nn.Linear(input_dim, output_dim//4),
                nn.GELU(),
                nn.Linear(output_dim//4, output_dim//4),
                nn.GELU(),
                nn.Linear(output_dim//4, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim) #增加一项全连接层
            ))
    
    def forward(self, task_embed: torch.Tensor):
        task_adapter_embeddings = []
        for i in range(self.num_layers):
            task_adapter_embeddings.append(self.task_adapter_mlp_list[i](task_embed[i]),dim=0) # what if we do not give it mean[task_num, dim]
        
        return task_adapter_embeddings



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

class MaskDecoder_task(nn.Module):
    def __init__(
            self,
            MaskDecoder: nn.Module,
    ):
        super().__init__()
        self.MaskDecoder = MaskDecoder
    
    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            task_specific_embed: torch.Tensor,
    ):  
        global_attn_num = image_embeddings.shape[0]  #[n, b, ?, ?, ?]
        global_masks = []
        global_iou_pred = []
        for i in range(global_attn_num):
            if i == global_attn_num -1:
                concat = False
            else: 
                concat = True
            masks, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                task_specific_embed = task_specific_embed[i],
                concat = concat,  # 决定是否要将task_specific_embed进行concat
            )
            global_masks.append(masks)
            global_iou_pred.append(iou_pred)
        
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # return masks, iou_pred
        return torch.stack(global_masks), torch.stack(global_iou_pred)

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        task_specific_embed: torch.Tensor, # 加入可学习部分
        concat : bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # mask_tokens = self.MaskDecoder.mask_tokens.weight + task_specific_embed 
        # 虽然这里self.mask_tokens会因为数量变化了被随机初始化，但仍然加了一个task_specific_embed
        # 表示与前面的关系
        # mask_tokens = self.MaskDecoder.mask_tokens.weight + task_specific_embed
        output_tokens = torch.cat([self.MaskDecoder.iou_token.weight, self.MaskDecoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        mask_tokens = task_specific_embed.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        if concat:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings,mask_tokens), dim=1)
        else:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.MaskDecoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.MaskDecoder.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        # print(src.shape)
        upscaled_embedding = self.MaskDecoder.output_upscaling(src)
        # print(upscaled_embedding.shape)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.MaskDecoder.num_mask_tokens):
            hyper_in_list.append(self.MaskDecoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings
        # print(masks.shape)

        # Generate mask quality predictions
        iou_pred = self.MaskDecoder.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    
class U_decoder(nn.Module):
    def __init__(
            self,
            output_dim : int,
            global_attn_num : int
    ):
        super().__init__()
        
        self.u_fusion_list = nn.ModuleList()
        for i in range(global_attn_num):
            u_fusion = nn.Sequential(
                    nn.Conv2d(
                        output_dim * 2,
                        output_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    LayerNorm2d(output_dim))
            self.u_fusion_list.append(u_fusion)
    
    def forward(self, decoder_embeddings):
        for i in range(len(self.u_fusion_list)):
            if i == 0:
                raw = torch.concat([decoder_embeddings[i], decoder_embeddings[i+1]], dim=-1)
            else:
                raw = torch.concat([raw, decoder_embeddings[i+1]], dim = -1)

            raw = self.u_fusion_list[i](raw)
        

        return raw


class Sam_task(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        sam_model: Sam
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
        self.global_attn_num = len(sam_model.image_encoder.global_attn_indexes)

        self.task_adapter = Task_adapter(decoder_dim, image_encoder_dim//4, image_encoder_dim, self.global_attn_num)
        self.u_decoder = U_decoder(decoder_dim, self.global_attn_num)
        
        self.task_specific_embed_list = nn.ParameterList()
        self.mask_adapter_list = nn.ModuleList()
        self.image_neck_list = nn.ModuleList()
        for layer_i , blk in enumerate(sam_model.image_encoder.blocks):
            if layer_i in sam_model.image_encoder.global_attn_indexes:
                blk.attn = Attention_task(blk.attn)
                sam_model.image_encoder.blocks[layer_i] = Block_task(blk)

                # image_encoder_neck
                neck = nn.Sequential(
                    nn.Conv2d(
                        image_encoder_dim,
                        decoder_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    LayerNorm2d(decoder_dim),
                    nn.Conv2d(
                        decoder_dim,
                        decoder_dim,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    LayerNorm2d(decoder_dim),
                )
                self.image_neck_list.append(neck)


                # task_adapter
                task_specific_embed = torch.empty_like(sam_model.mask_decoder.mask_tokens.weight) #[task_num, decoder_embed]
                nn.init.normal_(task_specific_embed, std=0.02)
                task_specific_embed = nn.Parameter(task_specific_embed)
                self.task_specific_embed_list.append(task_specific_embed)

                #mask_decoder
                mask_adapter = nn.Sequential(
                    nn.Linear(decoder_dim, decoder_dim//4),
                    nn.GELU(),
                    nn.Linear(decoder_dim//4, decoder_dim),
                    nn.GELU(),
                    nn.Linear(decoder_dim, decoder_dim),
                )
                self.mask_adapter_list.append(mask_adapter)


        sam_model.image_encoder = ImageEncoderViT_task(sam_model.image_encoder)
        sam_model.mask_decoder = MaskDecoder_task(sam_model.mask_decoder)
        
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

        # task_embed preprocess + image_encoder
        task_embed = self.task_adapter(self.task_specific_embed_list)
        image_embeddings = self.sam.image_encoder(input_images, task_embed)
        
        # image_neck process
        for i in range(self.global_attn_num):
            image_embeddings[i] = self.image_neck_list(image_embeddings[i])

        # prompt encoder
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None,
        ) #[batch, 256, 32, 32]

        # hyper_mask_adapter
        mask_tokens = []
        for i in range(self.global_attn_num):
            mask_tokens.append(self.mask_adapter_list[i](self.task_specific_embed_list[i]))

        # mask_tokens = self.mask_adapter(self.task_specific_embed)
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            task_specific_embed = mask_tokens,
        )

        # u-type postprocess
        masks = self.u_decoder(
            decoder_embeddings = low_res_masks
        )

        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        # print(low_res_masks.shape)
        return outputs
    
    def init_weights(self):
        task_adapter = self.task_adapter.task_adapter_mlp_list
        layers = len(task_adapter)
        for layer in range(layers):
            nn.init.constant_(task_adapter[layer][-1].weight, 0)
            nn.init.constant_(task_adapter[layer][-1].bias, 0)
            
            #init the mask_adapter
            nn.init.constant_(self.mask_adapter_list[layer][-1].weight,0)
            nn.init.constant_(self.mask_adapter_list[layer][-1].bias,0)
        

    def save_parameters(self, filename: str) ->None:
        
        assert filename.endswith(".pt") or filename.endswith('.pth')
        num_task = self.global_attn_num
        task_embed_tensors = {f"task_specific_embed_{i:03d}": self.task_specific_embed_list[i].weight for i in range(num_task)}

        
        task_adapter_tensors = {}
        u_decoder_tensors = {}
        mask_decoder_tensors = {}
        mask_adapter_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        

        for key, value in state_dict.items():
            if 'task_specific_embed_list' in key:
                task_embed_tensors[key] = value
            if 'u_decoder' in key:
                u_decoder_tensors[key] = value
            if 'task_adapter' in key:
                task_adapter_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if 'mask_adapter_list' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**task_embed_tensors, **task_adapter_tensors, **u_decoder_tensors, **mask_decoder_tensors, **mask_adapter_tensors}
        torch.save(merged_dict, filename)
    
    def load_parameters(self, filename: str) -> None:
        
        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load task_specific_embed
        for i, task_embed in enumerate(self.task_specific_embed_list):
            saved_key = f"task_specific_embed{i:03d}"
            saved_tensor = state_dict[saved_key]
            task_embed.weight = nn.Parameter(saved_tensor)
        
        # load task_adapter
        task_adapter_keys = [k for k in sam_keys if 'task_adapter' in k]
        task_adapter_values = [state_dict[k] for k in task_adapter_keys]
        task_adapter_state_dict = {k:v for k, v in zip(task_adapter_keys, task_adapter_values)}
        sam_dict.update(task_adapter_state_dict)

        # load u_decoder
        u_decoder_keys = [k for k in sam_keys if 'u_decoder' in k]
        u_decoder_values = [state_dict[k] for k in u_decoder_keys]
        u_decoder_state_dict = {k:v for k, v in zip(u_decoder_keys, u_decoder_values)}
        sam_dict.update(u_decoder_state_dict)

        # load mask_decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_state_dict = {k:v for k,v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_state_dict)

        #load mask_adapter
        mask_adapter_keys = [k for k in sam_keys if 'mask_adapter_list' in k]
        mask_adapter_values = [state_dict[k] for k in mask_adapter_keys]
        mask_adapter_state_dict = {k:v for k,v in zip(mask_adapter_keys, mask_adapter_values)}
        sam_dict.update(mask_adapter_state_dict)

        self.sam.load_state_dict(sam_dict)



    

    
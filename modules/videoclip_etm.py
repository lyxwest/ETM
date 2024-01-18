import torch
from torch import nn
from collections import OrderedDict

import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, need_attn_mask_self: bool, need_attn_mask_cross: bool):
        super().__init__()
        self.transformer_heads = n_head
        self.attn_self = nn.MultiheadAttention(d_model, n_head)    #d_model:total dimension of the model
        self.attn_cross = nn.MultiheadAttention(d_model, n_head)    #d_model:total dimension of the model
        self.ln_1_q = LayerNorm(d_model)
        self.ln_1_kv = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.mlp_kv = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2_q = LayerNorm(d_model)
        self.ln_2_kv = LayerNorm(d_model)
        
        self.cross_ln_1_q = LayerNorm(d_model)
        self.cross_ln_1_kv = LayerNorm(d_model)
        self.cross_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.cross_ln_2 = LayerNorm(d_model)
        
        self.need_attn_mask_self = need_attn_mask_self
        self.need_attn_mask_cross = need_attn_mask_cross


    def build_attention_mask(self,transformer_heads, batch, num_segs):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(num_segs, num_segs)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(dim=0).expand(batch*transformer_heads, -1, -1)
        return mask

    def attention_self(self, q: torch.Tensor):
        t, b, _ = q.size()
        if self.need_attn_mask_self:
            attn_mask = self.build_attention_mask(transformer_heads=self.transformer_heads, batch=b ,num_segs=t)
            attn_mask = attn_mask.to(dtype=q.dtype, device=q.device)
        else:
            attn_mask = None
        return self.attn_self(q, q, q, need_weights=False, attn_mask=attn_mask)[0]    # x x x对应q k v

    
    def attention_cross(self, q: torch.Tensor, kv: torch.Tensor):
        t, b, _ = q.size()
        if self.need_attn_mask_cross:
            attn_mask = self.build_attention_mask(transformer_heads=self.transformer_heads, batch=b ,num_segs=t)
            attn_mask = attn_mask.to(dtype=q.dtype, device=q.device)
        else:
            None
        return self.attn_cross(q, kv, kv, need_weights=False, attn_mask=attn_mask)[0]    # x x x对应q k v

    def forward(self, qkv):
        if type(qkv) is tuple:
            q, kv = qkv
            q = q + self.attention_self(self.ln_1_q(q))
            kv = kv + self.attention_self(self.ln_1_kv(kv))
            q = q + self.mlp(self.ln_2_q(q))
            kv = kv + self.mlp_kv(self.ln_2_kv(kv))

            q = q + self.attention_cross(self.cross_ln_1_q(q), self.cross_ln_1_kv(kv))
            q = q + self.cross_mlp(self.cross_ln_2(q))
 
            return (q, kv)


class SCLTransformer(nn.Module):
    def __init__(self, width, layers: int, heads: int, num_seg:int, num_shuffle: int, need_attn_mask_self: bool = False, need_attn_mask_cross: bool = True):
        super().__init__()
        self.num_shuffle = num_shuffle
        self.resblocks = nn.Sequential(*[ResidualCrossAttentionBlock(width, heads, need_attn_mask_self, need_attn_mask_cross) for _ in range(layers)])
        self.adaptive_fusion = nn.Linear(num_seg, 1, bias=False)
        
        
    def shuffle(self, x: torch.Tensor):
        f, _, _ = x.shape
        shuffled_index = torch.randperm(f)
        a = torch.arange(f)
        while (a.equal(shuffled_index)):
            shuffled_index = torch.randperm(f)
        return x[shuffled_index, :, :]

    def get_right_emb(self, q: torch.Tensor):
        q = self.resblocks((q,q))[0]
        q = q.permute(2, 1, 0)
        q = self.adaptive_fusion(q)
        q = q.permute(2, 1, 0)
        return q
    


    def forward(self, x: torch.Tensor):
        x_concat = []
        x_concat.append(self.get_right_emb(x))
        
        reverse_index = [7,6,5,4,3,2,1,0]
        x_shuffled = x[reverse_index, :, :]
        x_shuffled = self.resblocks((x, x_shuffled))[0]
        x_shuffled = x_shuffled.permute(2, 1, 0)
        x_shuffled = self.adaptive_fusion(x_shuffled)
        x_shuffled = x_shuffled.permute(2, 1, 0)
        x_concat.append(x_shuffled)

        reverse_index = [3,2,1,0,7,6,5,4]
        x_shuffled = x[reverse_index, :, :]
        x_shuffled = self.resblocks((x, x_shuffled))[0]
        x_shuffled = x_shuffled.permute(2, 1, 0)
        x_shuffled = self.adaptive_fusion(x_shuffled)
        x_shuffled = x_shuffled.permute(2, 1, 0)
        x_concat.append(x_shuffled)
          
        reverse_index = [3,2,5,4,7,6,1,0]
        x_shuffled = x[reverse_index, :, :]
        x_shuffled = self.resblocks((x, x_shuffled))[0]
        x_shuffled = x_shuffled.permute(2, 1, 0)
        x_shuffled = self.adaptive_fusion(x_shuffled)
        x_shuffled = x_shuffled.permute(2, 1, 0)
        x_concat.append(x_shuffled)
        
        for _ in range(0, self.num_shuffle-3):
            x_shuffled = self.shuffle(x)

            x_shuffled = self.resblocks((x, x_shuffled))[0]
            
            x_shuffled = x_shuffled.permute(2, 1, 0)
            x_shuffled = self.adaptive_fusion(x_shuffled)
            x_shuffled = x_shuffled.permute(2, 1, 0)

            x_concat.append(x_shuffled)

        x_concat = torch.cat(x_concat, dim=0)
        
        return x_concat


class video_header(nn.Module):
    def __init__(self, vid_head, clip_state_dict, num_seg, num_shuffle):
        super().__init__()
        self.vid_header = vid_head
        
        assert vid_head in ["None", "Transf"]

        if self.vid_header == "Transf":
            embed_dim = clip_state_dict["text_projection"].shape[1]
            context_length = clip_state_dict["positional_embedding"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
            self.scltransformer = SCLTransformer(width=embed_dim, layers=3, heads=transformer_heads, num_seg=num_seg, num_shuffle=num_shuffle)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def get_right_emb(self, x):
        _, t, _ = x.size()
        x = x.contiguous()
        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.scltransformer.get_right_emb(x)
            x = x.permute(1, 0, 2)  # LND -> NLD          
            x = x.type(x_original.dtype) + x_original.mean(dim=1, keepdim=True)
        return x.mean(dim=1, keepdim=False)


    def forward(self, x):
        _, t, _ = x.size()
        x = x.contiguous()
        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.scltransformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD            
            x = x.type(x_original.dtype) + x_original.mean(dim=1, keepdim=True)
            
        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        return x


            



class VideoCLIP(nn.Module):
    def __init__(self, clip_model, video_header: video_header, n_seg: int) :
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual
        self.fusion_model = video_header
        self.n_seg = n_seg
        self.logit_scale = clip_model.logit_scale

        self.tempreture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text_emb: torch.Tensor, list_id: torch.Tensor):
        video_emb = self.encode_video(image) 
        video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)

        #cross-entrophy loss
        video_right_emb = video_emb[:, 0, :].squeeze(dim=1)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        ce_logits = logit_scale * video_right_emb @ text_emb.t()

        #contrastive loss
        text_emb = self.get_target_emb(text_emb=text_emb, list_id=list_id)
        text_emb = text_emb.to(video_emb.device)
        contrastive_logits = torch.einsum("bnc, bc -> bn",[video_emb, text_emb])
        contrastive_logits /= self.tempreture
        return contrastive_logits, ce_logits


    def get_target_emb(self, text_emb: torch.Tensor, list_id: torch.Tensor):
        target_emb = text_emb[list_id]
        return target_emb


    def encode_right_video(self, image):
        bt = image.size(0)
        b = bt // self.n_seg
        image_emb = self.visual(image)
        image_emb = image_emb.view(b, self.n_seg, -1)
        video_emb = self.fusion_model.get_right_emb(image_emb)
        return video_emb


    def encode_video(self, image):
        bt = image.size(0)
        b = bt // self.n_seg
        image_emb = self.visual(image)
        if image_emb.size(0) == b:
            return image_emb
        else:
            image_emb = image_emb.view(b, self.n_seg, -1)    #(batchsize, n_seg, c)   
            image_emb = self.fusion_model(image_emb)  
            return image_emb
import torch
import torch.nn as nn
from ttnn import TTNNModel, TTNNLayer, TTNNConfig
from ttnn.layers import Linear, Conv2d, LayerNorm, MultiheadAttention

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Linear(dim, mlp_hidden_dim),
            act_layer(),
            Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class MaskFormerSwinB(TTNNModel):
    def __init__(self, config: TTNNConfig):
        super().__init__(config)
        self.patch_embed = PatchEmbed(
            img_size=config.img_size, patch_size=config.patch_size, in_chans=config.in_chans, embed_dim=config.embed_dim,
            norm_layer=LayerNorm
        )
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        self.layers = nn.ModuleList([
            SwinTransformerBlock(
                dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale, drop=config.drop_rate, attn_drop=config.attn_drop, drop_path=config.drop_path[i],
                act_layer=nn.GELU, norm_layer=LayerNorm
            )
            for i in range(config.num_layers)
        ])

        self.norm = LayerNorm(config.embed_dim)
        self.head = Linear(config.embed_dim, config.num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = TTNNConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        state_dict = torch.load(pretrained_model_name_or_path)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        torch.save(self.state_dict(), save_directory)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
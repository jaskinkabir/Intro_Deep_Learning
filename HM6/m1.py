from jlib.vision_transformer import VisionTransformer
from run_model import train_model
device = 'cuda:0'

vit = VisionTransformer(
    image_size=32,
    patch_size=4,
    embed_dim=192,
    inner_dim=384,
    num_attn_heads=6,
    num_attn_layers=12,
    num_classes=100,
    dropout=0.2,
    cls_head_dims=[384,192],
    device = device
)
train_model(
    model=vit,
    model_name='192-384',
    chart_title='ViT Emb:192 Inner:384',
    epochs=50,
    device=device
)
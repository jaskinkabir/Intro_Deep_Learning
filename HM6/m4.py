from jlib.vision_transformer import VisionTransformer
from run_model import train_model
device = 'cuda:3'

vit = VisionTransformer(
    image_size=32,
    patch_size=4,
    embed_dim=384,
    inner_dim=1536,
    num_attn_heads=6,
    num_attn_layers=12,
    num_classes=100,
    dropout=0.2,
    cls_head_dims=[384,192],
    device=device
)

train_model(
    model=vit,
    model_name='384-1536',
    chart_title='ViT Emb:384 Inner:1536',
    epochs=50,
    device=device
)
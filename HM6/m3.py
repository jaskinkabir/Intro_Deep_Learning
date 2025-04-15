from jlib.vision_transformer import VisionTransformer
from run_model import train_model
device = 'cuda:2'

vit = VisionTransformer(
    image_size=32,
    patch_size=8,
    embed_dim=256,
    inner_dim=512,
    num_attn_heads=4,
    num_attn_layers=4,
    num_classes=100,
    dropout=0.2,
    cls_head_dims=[256,128],
    device=device,
)

train_model(
    model=vit,
    model_name='8p256e',
    chart_title='ViT Patch:8 Emb:256',
    epochs=50,
    device=device
)
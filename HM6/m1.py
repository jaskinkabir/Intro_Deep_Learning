from jlib.vision_transformer import VisionTransformer, History
from run_model import train_model

vit = VisionTransformer(
    image_size=32,
    patch_size=4,
    embed_dim=256,
    inner_dim=512,
    num_attn_heads=4,
    num_attn_layers=8,
    num_classes=100,
    dropout=0.3,
    cls_head_dims=[1024,128]
)

train_model(
    model=vit,
    model_name='vit1',
    chart_title='ViT Model 1',
    epochs=50,
    device='cuda'
)
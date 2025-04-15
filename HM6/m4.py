from jlib.vision_transformer import VisionTransformer
from run_model import train_model
device = 'cuda:3'

vit = VisionTransformer(
    image_size=32,
    patch_size=8,
    embed_dim=512,
    inner_dim=1024,
    num_attn_heads=4,
    num_attn_layers=4,
    num_classes=100,
    dropout=0.2,
    cls_head_dims=[384,192],
    device=device,
)

train_model(
    model=vit,
    model_name='8p512e',
    chart_title='ViT Patch:8 Emb:384',
    epochs=50,
    device=device
)
from jlib.vision_transformer import VisionTransformer
from run_model import train_model

device = 'cuda:3'

model = VisionTransformer(
    image_size=32,
    patch_size=8,
    embed_dim=192,
    inner_dim=768,
    num_attn_heads=8,
    num_attn_layers=16,
    num_classes=100,
    dropout=0.2,
    cls_head_dims=[384,192],
)

train_model(
    model=model,
    model_name='vit4',
    chart_title='ViT Model 4',
    epochs=50,
    device=device,
)
from jlib.vision_transformer import VisionTransformer
from run_model import train_model

model = VisionTransformer(
    image_size=32,
    patch_size=4,
    embed_dim=192,
    inner_dim=384,
    num_attn_heads=6,
    num_attn_layers=12,
    num_classes=100,
    dropout=0.2,
    cls_head_dims=[384,192],
)

train_model(
    model=model,
    model_name='cifar2',
    chart_title='ViT Model 2',
    epochs=100,
)
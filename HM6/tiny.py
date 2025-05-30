from jlib.swin import Swin
from run_swin import train_model
device = 'cuda'

model = Swin(model_name='microsoft/swin-tiny-patch4-window7-224', num_classes=100, device=device)
train_model(
    model=model,
    swin='microsoft/swin-tiny-patch4-window7-224',
    model_name='swin_tiny',
    chart_title='Swin Tiny',
    epochs=5,
    lr=1e-3,
    device=device,
)

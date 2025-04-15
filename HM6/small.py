from jlib.swin import Swin
from run_swin import train_model
device = 'cuda:1'

model = Swin(model_name='microsoft/swin-small-patch4-window7-224', num_classes=100, device=device)
train_model(
    model=model,
    swin='microsoft/swin-small-patch4-window7-224',
    model_name='swin_small',
    chart_title='Swin Small',
    epochs=5,
    lr=2e-5,
    device=device,
)
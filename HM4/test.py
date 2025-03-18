from data import english_to_french
from jlib.get_enfr_loader import EnglishToFrench, Language, get_enfr_loader

en2fr = EnglishToFrench(english_to_french)
loaders = get_enfr_loader(
    english_to_french,
    train_batch_size=16,
    val_batch_size=16,
    workers=12,
)
print('done')
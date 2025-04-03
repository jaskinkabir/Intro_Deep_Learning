from .classifier_no_data_loaders import ClassifierNoDataLoaders
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class CharPredictor(ClassifierNoDataLoaders):
    def __init__(self):
        super().__init__()
    def pass_text(self, text: str, sequence_length: int) -> None:
        self.text = text
        self.sequence_length = sequence_length

        # Creating character vocabulary
        # part of the data preprocessing step for a character-level text modeling task. 
        # Create mappings between characters in the text and numerical indices

        #set(text): Creates a set of unique characters found in the text. The set function removes any duplicate characters.
        #list(set(text)): Converts the set back into a list so that it can be sorted. 
        # sorted(list(set(text))): Sorts the list of unique characters. 
        chars = sorted(list(set(text)))
        #This line creates a dictionary that maps each character to a unique index (integer)."
        ix_to_char = {i: ch for i, ch in enumerate(chars)}
        #Similar to the previous line, but in reverse. This line creates a dictionary that maps each unique index (integer) back to its corresponding character.
        char_to_ix = {ch: i for i, ch in enumerate(chars)} 
        chars = sorted(list(set(text)))
        
        self.input_size = len(chars)
        self.output_size = len(chars)
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char
        self.chars = chars
        
        # Preparing the dataset
        x = []
        y = []
        for i in range(len(text) - sequence_length):
            sequence = text[i:i + sequence_length]
            label = text[i + sequence_length]
            x.append([char_to_ix[char] for char in sequence])
            y.append(char_to_ix[label])

        x = np.array(x)
        y = np.array(y)

        # Splitting the dataset into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

        # Converting data to PyTorch tensors
        self.x_train = torch.tensor(x_train, dtype=torch.long)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.x_val = torch.tensor(x_val, dtype=torch.long)
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        
        
    def train_model(self, *args, **kwargs):
        kwargs['x_train'] = self.x_train
        kwargs['y_train'] = self.y_train
        kwargs['x_val'] = self.x_val
        kwargs['y_val'] = self.y_val
        
        super().train_model(*args, **kwargs)

        
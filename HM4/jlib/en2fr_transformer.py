class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1 : '<SOS>', 2: "<EOS>"}
        self.n_words = 3
        self.max_sentence_length = 0
        
    def add_sentence(self, sentence):
        split = sentence.split(' ')
        if len(split) > self.max_sentence_length:
            self.max_sentence_length = len(split)
        for word in split:
            self.add_word(word)
            
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Load data (English to French)
class EnglishToFrench(Dataset):
    
    def __init__(self, en2fr, max_length, gpu=False):
        self.sequences = []
        self.targets = []
        self.en = Language('English')
        self.fr = Language('French')
        
        for english, french in en2fr:
            self.en.add_sentence(english)
            self.fr.add_sentence(french)
        
        self.max_length = max(max_length, self.en.max_sentence_length, self.fr.max_sentence_length) + 2

        for english, french in en2fr:
            en_seq = self.sentence_to_sequence(english, self.en)
            self.sequences.append(en_seq)
            
            fr_seq = self.sentence_to_sequence(french, self.fr)
            self.targets.append(fr_seq)
            
        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        if gpu:
            self.sequences = self.sequences.cuda()
            self.targets = self.targets.cuda()
        
    def sentence_to_sequence(self, sentence, language):
        seq = [SOS]
        for word in sentence.split():
            seq.append(language.word2index[word])
        
        if len(seq) < self.max_length-1:
            seq += [PAD] * (self.max_length - len(seq) - 1)
        
        seq.append(EOS)
        return seq
            
            
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
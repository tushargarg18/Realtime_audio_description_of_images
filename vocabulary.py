import collections
import re
import json
import yaml

#--------------------------------------------------------------------------------------
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "<unk>": 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.next_index = 4

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenizer(text):
        # very simple whitespace+punctuation splitter
        text = text.strip().lower()
        # split on any non-word character
        return [t for t in re.split(r"\W+", text) if t]

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for token, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.word2idx[token] = self.next_index
                self.idx2word[self.next_index] = token
                self.next_index += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        numericalized = [self.word2idx["<start>"]]
        for token in tokenized_text:
            idx = self.word2idx.get(token, self.word2idx["<unk>"])
            numericalized.append(idx)
        numericalized.append(self.word2idx["<end>"])
        return numericalized
    
    def from_json(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
            self.idx = len(self.word2idx)  # or data.get("idx", len(self.word2idx))
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
# Read the configurations
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

captions_dataset = config["data"]["captions_file"]
freq_threshold = config["training"]["vocab_freq"]
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
# Read the captions file from the dataset  
with open(str(captions_dataset), "r", encoding="utf-8") as f:
        captions = [str(line.strip().lower().split(',')[1]) for line in f.readlines()]

vocab = Vocabulary(freq_threshold = 5)
vocab.build_vocabulary(captions)

# Write the generated vocabulary to generate a json file
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump({
        "word2idx": vocab.word2idx,
        "idx2word": vocab.idx2word
    }, f, ensure_ascii = False, indent = 4)
#--------------------------------------------------------------------------------------
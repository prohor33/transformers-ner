from transformers import AutoTokenizer
from tqdm import tqdm

model_name = "bert-base-cased"

class BERTDataset():
    def __init__(self, model_name, sequences, label_to_index={}):
        self.read_only_label_to_index = bool(label_to_index)
        self.label_to_index = label_to_index
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenized_sequences = []
        for seq in tqdm(sequences):
            tokenized_sequences.append(self.tokenize_and_align(*seq))
        self.sequences = tokenized_sequences
    
    def convert_label_to_index(self, label):
        if label != "O":
            label = label.split('-')[1]
        if not label in self.label_to_index:
            if self.read_only_label_to_index:
                raise RuntimeError(f"No label \"{label}\" in supplyed label_to_index: {self.label_to_index}.")
            self.label_to_index[label] = len(self.label_to_index)
        return self.label_to_index[label]
        
    def tokenize_and_align(self, tokens, tags):
        output = self.tokenizer(tokens, padding=False, truncation=True, is_split_into_words=True)
        
        word_ids = output.word_ids(batch_index=0)
        
        labels_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx in (None, prev_word_idx):
                # setting label only for first token of each word
                labels_ids.append(-100)
            else:
                labels_ids.append(self.convert_label_to_index(tags[word_idx]))
            prev_word_idx = word_idx
        
        output["labels"] = labels_ids
        return output
        
    def __iter__(self):
        for seq in self.sequences:
            yield seq
            
    def __getitem__(self, idx):
        return self.sequences[idx]
            
    def __len__(self):
        return len(self.sequences)
    
    def num_classes(self):
        return len(self.label_to_index)

    def idx_to_label(self):
        assert len(self.label_to_index.values()) == len(set(self.label_to_index.values()))
        return {v: k for k, v in self.label_to_index.items()}
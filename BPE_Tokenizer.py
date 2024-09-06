import regex as re

class BPETokenizer:
    def __init__(self, text, lang='en', iterations=3, min_merge_freq=5):
        self.text = text.lower()
        self.lang = lang
        self.iterations = iterations
        self.min_merge_freq = min_merge_freq
        self.stoi, self.itos = self.build_dict(self.text)
        
    def vocab_size(self):
        return len(stoi)
        
    def build_dict(self, text):
        unique_chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(unique_chars)}
        itos = {i: ch for i, ch in enumerate(unique_chars)}
        return stoi, itos

    def word_tokenize(self, text):
        if self.lang == 'en':
            word_tokenizer = re.compile(
                r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            )
            return re.findall(word_tokenizer, text)
        elif self.lang == 'ur':
            words = text.split()
            return [' ' + word for word in words]

    def convert_to_tokens(self, word):
        return [self.stoi[ch] for ch in word]

    def encode(self):
        words = self.word_tokenize(self.text)
        word_tokens = [self.convert_to_tokens(word) for word in words]

        for _ in range(self.iterations):
            pair_counts = self.get_pair_counts(word_tokens)
            merges = {
                k: v for k, v in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)
                if v >= self.min_merge_freq
            }

            self.update_tokens(merges)
            word_tokens = self.merge_tokens(word_tokens, merges)
        
        flattened_tokens = [token for word in word_tokens for token in word]
        return flattened_tokens

    def get_pair_counts(self, word_tokens):
        counts = {}
        for token_list in word_tokens:
            for pair in zip(token_list, token_list[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def update_tokens(self, merges):
        for (char1, char2) in merges:
            new_token = self.itos[char1] + self.itos[char2]
            if new_token not in self.stoi:
                self.stoi[new_token] = len(self.stoi)
                self.itos[len(self.itos)] = new_token

    def merge_tokens(self, word_tokens, merges):
        new_word_tokens = []
        for word in word_tokens:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) in merges:
                    new_word.append(self.stoi[self.itos[word[i]] + self.itos[word[i+1]]])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tokens.append(new_word)
        return new_word_tokens

    def decode(self, tokens):
        return ''.join([self.itos[token] for token in tokens])

# Main
import pandas as pd
text_list = pd.read_csv('Sentiment Dataset Urdu.csv')['Text']
text = ''
for line in text_list:
    text += line
tokenizer = BPETokenizer(text,lang='ur')
encoded_tokens = tokenizer.encode()
decoded_text = tokenizer.decode(encoded_tokens)

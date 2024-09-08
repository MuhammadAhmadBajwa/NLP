import regex as re

class TrieNode:
    def __init__(self):
        self.children = {}
        self.token_id = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, token_id):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_id = token_id

    def search_longest(self, text, start_index):
        node = self.root
        longest_match = None
        longest_match_len = 0

        for i in range(start_index, len(text)):
            char = text[i]
            if char in node.children:
                node = node.children[char]
                if node.token_id is not None:
                    longest_match = node.token_id
                    longest_match_len = i - start_index + 1
            else:
                break

        return longest_match, longest_match_len


class BPETokenizer:
    def __init__(self, text, iterations=4, min_merge_freq=0):
        self.text = text.lower()
        self.iterations = iterations
        self.min_merge_freq = min_merge_freq
        self.stoi, self.itos = self.build_dict(self.text)
        self.stoi["<UNK>"] = len(self.stoi)
        self.itos[len(self.itos)] = "<UNK>"
    def vocab_size(self):
        return len(self.stoi)

    def build_dict(self, text):
        unique_chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(unique_chars)}
        itos = {i: ch for i, ch in enumerate(unique_chars)}
        return stoi, itos

    def word_tokenize(self, text):
        word_tokenizer = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        return re.findall(word_tokenizer, text)

    def convert_to_tokens(self, word):
        return [self.stoi[ch] for ch in word]

    def train(self):
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

    def encode(self,text):
        trie = Trie()
        for word, token_id in self.stoi.items():
            trie.insert(word, token_id)

        tokens = []
        i = 0

        while i < len(text):
            # Find the longest match in the Trie starting from index i
            token_id, match_len = trie.search_longest(text, i)

            if token_id is not None:
                tokens.append(token_id)
                i += match_len
            else:
                tokens.append(self.stoi["<UNK>"])  # <UNK> represents unknown tokens
                i += 1

        return tokens

    def decode(self, tokens):
        return ''.join([self.itos[token] for token in tokens])

# # Main
# text = open('/content/TheVerdict.txt','r',encoding='utf-8').read().lower()
# tokenizer = BPETokenizer(text)
# tokenizer.train()
# encoded_tokens = tokenizer.encode('hello world')
# print(encoded_tokens)
# decoded_text = tokenizer.decode(encoded_tokens)
# print(decoded_text)

# # Main
# import pandas as pd
# text_list = pd.read_csv('Sentiment Dataset Urdu.csv',encoding='utf-8')['Text']
# text = ''
# for line in text_list:
#     text += line
# tokenizer = BPETokenizer(text)
# encoded_tokens = tokenizer.encode()
# decoded_text = tokenizer.decode(encoded_tokens)

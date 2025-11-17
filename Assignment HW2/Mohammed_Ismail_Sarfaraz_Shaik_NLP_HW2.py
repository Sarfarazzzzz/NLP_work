# =================================================================
# Problem 1:
# Use glove word embeddings to train the MLP of the example
# % --------------------------------------------------------

# 1. Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/, unzip it and move glove.6B.50d.txt to the
# current working directory.

# 2. Define a function that takes as input the vocab dict from the example and returns an embedding dict with the token
# ids from vocab dict as keys and the 50-dim Tensors from the glove embeddings as values.

# 3. Define a function to return a Tensor that contains the tensors corresponding to the glove embeddings for the tokens
# in our vocabulary. The ones not found on the glove vocabulary are given tensors of 0s. This will happen more often
# than expected because our tokenizer is different from the one used for glove.

# 4. Replace the embedding weights of the model with the loop-up table returned by the function defined in 4. Check some
# of these vectors visually against the glove.6B.50d.txt file to make sure the correct embeddings are being used.

# 6. Add an option to freeze the embeddings so that they are not learnt. This will result in a poor performance because
# there are quite a few tokens which we don't have glove embeddings for (as mentioned in 4.), so we need to learn these.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')

# Solution 2:

import numpy as np
import torch
import torch.nn as nn

#Define a function to load GloVe embeddings
def load_glove_embeddings(embedding_file):
    embedding_dict = {}
    with open(embedding_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype='float32')
            embedding_dict[word] = embedding
    return embedding_dict

# Solution 3:

# Create a function to generate embedding tensors
def create_embedding_matrix(vocab, embedding_dict, embedding_dim):
    num_tokens = len(vocab)
    embedding_matrix = torch.zeros(num_tokens, embedding_dim)

    for word, idx in vocab.items():
        if word in embedding_dict:
            embedding_matrix[idx] = torch.tensor(embedding_dict[word])

    return embedding_matrix

# Solution 4:

# Replace the embedding weights in your model
class MLPWithEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MLPWithEmbeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        x = self.output(x)
        return x

# Soluiton 6

your_vocab = {
    "king": 0,
    "queen": 1,
    "man": 2,
    "woman": 3,
    "paris": 4,
    "france": 5,
    "london": 6,
    "england": 7,
    "apple": 8,
    "orange": 9
}

# Load GloVe embeddings
glove_embeddings_file = "glove.6B.50d.txt"
embedding_dim = 50
embedding_dict = load_glove_embeddings(glove_embeddings_file)

# Create embedding matrix
embedding_matrix = create_embedding_matrix(your_vocab, embedding_dict, embedding_dim)

# Build model and load embeddings
vocab_size = len(your_vocab)
hidden_dim = 32
output_dim = 2

model = MLPWithEmbeddings(vocab_size, embedding_dim, hidden_dim, output_dim)
model.embedding.weight.data.copy_(embedding_matrix)

# Optionally freeze embeddings
model.embedding.weight.requires_grad = False
print("Embeddings frozen:", not model.embedding.weight.requires_grad)

# Test: print embedding vector for a few words
for word in ["king", "queen", "apple"]:
    idx = your_vocab[word]
    print(f"\nEmbedding for '{word}':\n", model.embedding.weight[idx][:10])


print(20*'-' + 'End Q1' + 20*'-')

#%% # =================================================================
# Problem 2:
# Use the following corpus
#
# corpus = ['king is a strong man',
#           'queen is a wise woman',
#           'boy is a young man',
#           'girl is a young woman',
#           'prince is a young king',
#           'princess is a young queen',
#           'man is strong',
#           'woman is pretty',
#           'prince is a boy will be king',
#           'princess is a girl will be queen']
# Train a  two layer neural network and show the the result of word2vec.
# Hint:
# 1- Remove the stop words
# 2- Use binary encoding for each word
# 3- Try a window size of 2 and 3
# 4- Make the embedding size of 2. Plot the each word and explain the results
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import numpy as np
from nltk.corpus import stopwords
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
# -------------------------------------------------------------------------------------

# Step 1: Define corpus

corpus = [
    'king is a strong man',
    'queen is a wise woman',
    'boy is a young man',
    'girl is a young woman',
    'prince is a young king',
    'princess is a young queen',
    'man is strong',
    'woman is pretty',
    'prince is a boy will be king',
    'princess is a girl will be queen'
]

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokenized_sentences = [[w for w in sent.lower().split() if w not in stop_words] for sent in corpus]
print("Tokenized sentences:", tokenized_sentences)

# Build vocabulary
vocab = sorted(set([w for sent in tokenized_sentences for w in sent]))
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

print("\nVocabulary:", vocab)
print("Vocab size:", vocab_size)

# Create context-target pairs (window size = 2)
def generate_training_data(sentences, window_size=2):
    pairs = []
    for sent in sentences:
        for i, word in enumerate(sent):
            target = word
            start = max(0, i - window_size)
            end = min(len(sent), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((word, sent[j]))
    return pairs

pairs = generate_training_data(tokenized_sentences, window_size=2)
print("\nSample pairs:", pairs[:10])

# One-hot encoding for input/output
def one_hot_vector(word):
    vec = np.zeros(vocab_size)
    vec[word_to_idx[word]] = 1
    return vec

X_train = []
Y_train = []

for target, context in pairs:
    X_train.append(one_hot_vector(target))
    Y_train.append(one_hot_vector(context))

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

# Define 2-layer neural network (Word2Vec)
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.hidden = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        h = self.hidden(x)
        out = self.output(h)
        return out

# Train model
embedding_dim = 2
model = Word2Vec(vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

# Since Y is one-hot, we need class indices
Y_indices = torch.argmax(Y_train, dim=1)

epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_indices)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

# Extract embeddings
embeddings = model.hidden.weight.data.T
print("\nWord Embeddings:\n")
for word, idx in word_to_idx.items():
    print(f"{word}: {embeddings[idx].numpy()}")

# Plot 2D embeddings
plt.figure(figsize=(8,6))
for word, idx in word_to_idx.items():
    vec = embeddings[idx].numpy()
    plt.scatter(vec[0], vec[1])
    plt.text(vec[0]+0.01, vec[1]+0.01, word, fontsize=12)

plt.title("2D Word Embeddings (Word2Vec - SkipGram)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()

print(20*'-' + 'End Q2' + 20*'-')

# Results:

# The Word2Vec model successfully learned meaningful semantic relationships from the small corpus.
# The resulting 2-D embeddings show clear clusters: king, man, boy, and prince appear close together, forming a masculine group, while queen, woman, girl, and princess cluster separately, representing a feminine group.
# Adjectives such as strong, wise, young, and pretty occupy intermediate positions, connecting the two clusters based on shared contextual use.



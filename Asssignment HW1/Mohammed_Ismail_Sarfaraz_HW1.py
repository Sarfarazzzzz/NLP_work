#%%

import argparse
import os
import re
import pandas as pd
from collections import Counter

from nltk.book import text1
from nltk.corpus import gutenberg
from nltk import FreqDist, pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import requests
from bs4 import BeautifulSoup
import random

from nltk.corpus import words as nltk_words
from nltk.stem import PorterStemmer


#%%

# PART A

'''- Pull the sample.txt file from the class Github repository and unzip it. Note this assignment is intended to deepen your learning of using argparse for parsing command-line arguments. 
- Write your Python script in a way that I can run it with parser in the terminal. The sample.txt is the argument for the parser.
Here's beginning of the code 
import argparse
import os
import pandas as pd
import re

#Split a text (string) into a list of sentences.
def extract_sentences(text):
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
return sentences
- The solution at the end should make a directory with the name of Text Feature. You need to read the sample.txt in your solution script file and then decompose the text to the sentence level by the basic rules. Save the results in the pandas dataframe which rows are the sentences and the column is the number of words in the sentence. Save the pandas dataframe in to csv file named sent.csv in to the text feature directory.'''

# Split text into sentences
def extract_sentences(text: str):
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    text = re.sub(r'\s+', ' ', text.strip())
    return re.split(pattern, text) if text else []

# Count words in a sentence
def count_words(sentence: str) -> int:
    return len(re.findall(r"\b[\w']+\b", sentence))

def main():
    parser = argparse.ArgumentParser(
        description="Process sample.txt → split into sentences → save sentence + word count CSV."
    )
    parser.add_argument("input_file", help="Path to sample_text.txt")
    parser.add_argument("--outdir", default="Text Feature",
                        help="Output directory (default: Text Feature)")
    args = parser.parse_args()

    # Read the text file
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract sentences
    sentences = [s.strip() for s in extract_sentences(text) if s.strip()]

    # Create DataFrame
    df = pd.DataFrame({
        "sentence": sentences,
        "num_words": [count_words(s) for s in sentences]
    })

    # output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Save CSV
    out_path = os.path.join(args.outdir, "sent.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Processed {len(df)} sentences.")
    print(f"Saved CSV to: {out_path}")

if __name__ == "__main__":
    main()


#%%

# PART B

'''
Part B.
For this part, you will u se NLTK to explore the Moby Dick text.
i. Analyzing Moby Dick text. Load the moby.txt file into python environment. (Load the raw data or Use the NLTK Text object)
ii. Tokenize the text into words. How many tokens (words and punctuation symbols) are in it?
iii. How many unique tokens (unique words and punctuation) does the text have?
iv. After lemmatizing the verbs, how many unique tokens does it have?
v. What is the lexical diversity of the given text input?
vi. What percentage of tokens is ’whale’or ’Whale’?
vii. What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
viii. What tokens have a length of greater than 6 and frequency of more than 160?
ix. Find the longest word in the text and that word’s length.
x. What unique words have a frequency of more than 2000? What is their frequency?
xi. What is the average number of tokens per sentence?
xii. What are the 5 most frequent parts of speech in this text? What is their frequency?
'''
# i) Load Moby Dick
tokens = list(text1)

# Total tokens
num_tokens = len(tokens)

# iii) Unique tokens
unique_tokens = set(tokens)
num_unique = len(unique_tokens)

# iv) Lemmatize verbs and count count unique tokens
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)
lemmatized_tokens = [
    lemmatizer.lemmatize(tok, pos='v') if tag.startswith('V') else tok
    for tok, tag in tagged
]
num_unique_after_verb_lemma = len(set(lemmatized_tokens))

# v) Lexical diversity
lex_div = num_unique / num_tokens if num_tokens else 0.0

# vi) % tokens that are 'whale'/'Whale'
whale_count = sum(1 for t in tokens if t.lower() == 'whale')
whale_pct = 100.0 * whale_count / num_tokens if num_tokens else 0.0

# vii) Top 20 most frequent tokens
fd = FreqDist(tokens)
top20 = fd.most_common(20)

# viii) Tokens with length > 6 and frequency > 160
long_and_freq = sorted([tok for tok, c in fd.items() if len(tok) > 6 and c > 160])

# ix) Longest alphabetic word & its length
alpha_tokens = [t for t in tokens if t.isalpha()]
longest_word = max(alpha_tokens, key=len) if alpha_tokens else None
longest_len = len(longest_word) if longest_word else 0

# x) Unique words with frequency > 2000
word_fd = FreqDist(t for t in tokens if t.isalpha())
very_freq_words = sorted([(w, c) for w, c in word_fd.items() if c > 2000],
                         key=lambda x: (-x[1], x[0]))

# xi) Average number of tokens per sentence
raw_text = gutenberg.raw('melville-moby_dick.txt')
sentences = sent_tokenize(raw_text)
avg_tokens_per_sentence = (sum(len(word_tokenize(s)) for s in sentences) / len(sentences)) if sentences else 0.0

# xii) 5 most frequent POS tags
tag_fd = FreqDist(tag for _, tag in tagged)
top5_pos = tag_fd.most_common(5)

print(f"ii) Total tokens: {num_tokens}")
print(f"iii) Unique tokens: {num_unique}")
print(f"iv) Unique tokens after verb lemmatization: {num_unique_after_verb_lemma}")
print(f"v) Lexical diversity: {lex_div:.6f}")
print(f"vi) % tokens that are 'whale'/'Whale': {whale_pct:.4f}%")
print("vii) Top 20 tokens (token, freq):")
for tok, c in top20: print(f"   {tok!r}: {c}")
print("viii) Tokens with len>6 and freq>160:")
print("   ", long_and_freq)
print(f"ix) Longest word & length: ({longest_word!r}, {longest_len})")
print("x) Words with freq>2000 (word, freq):")
for w, c in very_freq_words: print(f"   {w!r}: {c}")
print(f"xi) Avg tokens per sentence: {avg_tokens_per_sentence:.4f}")
print("xii) Top 5 POS tags (tag, freq):")
for tag, c in top5_pos: print(f"   {tag}: {c}")


#%%

# PART C

'''

Part C.
Lets get some text file from the Benjamin Franklin wiki page.
i. Write a function that scrape the web page and return the raw text file.
ii. Use BeautifulSoup to get text file and clean the html file.
iii. Write a function called unknown, which removes any items from this set that occur in the Words Corpus (nltk.corpus.words).
iv. Find a list of novel words.
v. Use the porter stemmer to stem all the items in novel words the go through the unknown function, saving the result as novel-stems.
vi. Find as many proper names from novel-stems as possible, saving the result as proper names.

'''

url = "https://en.wikipedia.org/wiki/Benjamin_Franklin"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
html = resp.text

# ii) Clean HTML to plain text
soup = BeautifulSoup(html, "html.parser")
for tag in soup(["script", "style", "noscript"]):
    tag.extract()

main = soup.select_one("#mw-content-text") or soup
clean_text = re.sub(r"\s+", " ", main.get_text(separator=" ")).strip()

# Tokenize into words
tokens = word_tokenize(clean_text)

# iii) Unknown function = words not in NLTK words corpus
wordset = set(w.lower() for w in nltk_words.words())
def unknown(words):
    return {w for w in words if w.lower() not in wordset}

# iv) Novel words
wordlike = [t for t in tokens if t.isalpha()]
novel_words = unknown(wordlike)

# v) Porter stemmer → stem novel words → filter again with unknown
ps = PorterStemmer()
stems = [ps.stem(w) for w in novel_words]
novel_stems = unknown(stems)

# vi) Proper names
tagged = pos_tag(tokens)
proper_names = set()
for tok, tag in tagged:
    if tag in ("NNP", "NNPS") and tok.isalpha():
        if ps.stem(tok) in novel_stems:
            proper_names.add(tok)

print("Novel words:", list(novel_words))
print("Novel stems:", list(novel_stems))
print("Proper names:", list(proper_names))

print(len(novel_words))
print(len(novel_stems))
print(len(proper_names))

#%%

# PART D

'''
Part D.
In part of this exercise, you will use the twitter data.
i. Load the data from the class repository and view the first few sentences.
ii. Split data into sentences using ”\n” as the delimiter.
iii. Tokenize sentences (split a sentence into a list of words). Convert all tokens into lower case so that words which are capitalized
iv. Split data into training and test sets.
v. Count how many times each word appears in the data.
'''

# i. Load the data from the class repository and view the first few sentences
filename = "twitter.txt"

with open(filename, "r", encoding="utf-8") as f:
    raw_data = f.read()

print("First few sentences:")
print("\n".join(raw_data.splitlines()[:5]))

# ii. Split data into sentences using '\n'
sentences = raw_data.split("\n")
print(f"\nTotal number of sentences: {len(sentences)}")

# iii. Tokenize each sentence into words, convert to lowercase
tokenized_sentences = []
for sent in sentences:
    tokens = word_tokenize(sent)
    tokens = [t.lower() for t in tokens]
    tokenized_sentences.append(tokens)

print("\nExample tokenized sentence:", tokenized_sentences[0])

# iv. Split data into training and test sets (80/20 split)
random.shuffle(tokenized_sentences)
split_index = int(0.8 * len(tokenized_sentences))
train_data = tokenized_sentences[:split_index]
test_data = tokenized_sentences[split_index:]

print(f"\nTraining set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# v. Count how many times each word appears in the data
all_tokens = [tok for sent in tokenized_sentences for tok in sent]
word_freq = Counter(all_tokens)

print("\nCount of Each word:")
for word, count in word_freq.most_common():
    print(f"{word}: {count}")





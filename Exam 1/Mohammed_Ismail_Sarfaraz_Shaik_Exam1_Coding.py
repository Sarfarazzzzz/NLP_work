#**********************************
#     DATS6312.11_Exam_1_FA25    *
#**********************************
#==========================================Part A======================================================================================
"""
In PART A, you are required to obtain the NPS chat corpus from the NLTK source and perform a com-
prehensive cleaning of each document, ensuring that all essential cleaning procedures are followed. Sub-
sequently, your task involves summarizing each document and generating a numerical representation of the
summarized content.
"""
import re
import nltk
from nltk.corpus import nps_chat, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy
from tqdm import tqdm
import pandas as pd
#Complete the rest of the code here:

posts = nps_chat.posts()
docs = [" ".join(p) for p in posts]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

cleaned_docs = [clean_text(doc) for doc in tqdm(docs, desc="Cleaning")]

def summarize_text(text, n=10):
    tokens = word_tokenize(text)
    freq = nltk.FreqDist(tokens)
    most_common = [w for w, _ in freq.most_common(n)]
    return " ".join(most_common)

summaries = [summarize_text(doc) for doc in tqdm(cleaned_docs, desc="Summarizing")]

vectorizer = TfidfVectorizer(max_features=300)
tfidf_vectors = vectorizer.fit_transform(summaries)

print("TF-IDF shape:", tfidf_vectors.shape)

df_partA = pd.DataFrame(tfidf_vectors.toarray(), columns=vectorizer.get_feature_names_out())
print(df_partA.head())


#%%
#==========================================Part B======================================================================================
"""
In Part B, you need to download a text data from the NLTK movie_reviews, which I already wrote 
the code for you. Use you knowledge that you learned in the class and clean the text appropriately.
After Cleaning is done, please find the numerical representation of text by any methods that you learned.
You need to find a creative way to label the sentiment of the sentences. The dataset already has positive and negative labels.
Labeling sentences as 'positive' or 'negative' based on sentiment scores and then evaluate your predicted sentiments.
Create a Pandas dataframe with sentences, true sentiment labels and predicted sentiment labels.
Calculate the accuracy of your predicted sentiment and true sentiments.
"""
#==============================================================================================================================================
#%%
#Load the libraries and packages you need for the analysis
from nltk.corpus import movie_reviews
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score

#%%
# Load movie reviews
neg_reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids("neg")]
pos_reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids("pos")]
#create a df of the reviews and sentiment labels
reviews = pd.DataFrame(
    {"text": neg_reviews + pos_reviews, "sentiment": ["negative"] * len(neg_reviews) + ["positive"] * len(pos_reviews)}
)
#%%
# Q.B-1. For text preprocessing, consider tokenization, lemmatization, and stop word removal

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

reviews["clean_text"] = reviews["text"].apply(preprocess)

# Q.B-2. Conduct TF-IDF Vectorization

tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(reviews["clean_text"])

# Initialize the VADER sentiment analyzer

sid = SentimentIntensityAnalyzer()
# Perform sentiment analysis and store sentiment scores in the DataFrame

reviews["sentiment_scores"] = reviews["text"].apply(lambda text: sid.polarity_scores(text))

# Q.B-3. Define a threshold for classifying sentences as positive or negative. Write a function to classify sentiment
# based on the compound score from VADER

def classify_sentiment(scores, threshold=0.05):
    compound = scores['compound']
    if compound >= threshold:
        return 'positive'
    elif compound <= -threshold:
        return 'negative'
    else:
        return 'positive' if scores['pos'] > scores['neg'] else 'negative'

# Apply with default threshold

reviews["predicted_sentiment"] = reviews["sentiment_scores"].apply(lambda x: classify_sentiment(x, threshold=0.05))

# Calculate accuracy

accuracy = accuracy_score(reviews["sentiment"], reviews["predicted_sentiment"])
print("Accuracy with threshold 0.05:", accuracy)

# Compare different thresholds and pick the best

thresholds = [0.0, 0.05]
best_acc = 0
best_threshold = 0.05

for t in thresholds:
    col_name = f"pred_{t}"
    reviews[col_name] = reviews["sentiment_scores"].apply(lambda x: classify_sentiment(x, threshold=t))
    acc = accuracy_score(reviews["sentiment"], reviews[col_name])
    print(f"Threshold {t}: Accuracy = {acc}")
    if acc > best_acc:
        best_acc = acc
        best_threshold = t

# Update final predicted sentiment based on best threshold

reviews["predicted_sentiment"] = reviews["sentiment_scores"].apply(lambda x: classify_sentiment(x, threshold=best_threshold))

# Display final DataFrame

df = reviews[["text", "sentiment", "predicted_sentiment"]]

for i, row in df.head(5).iterrows():
    text_preview = row['text'][:200]
    print(f"Review {i+1}: {text_preview}...")
    print(f"Predicted sentiment: {row['predicted_sentiment']}")
    print('-'*60)

#============================PART C=======================================================================================================
# This dataset is about classifing a given sample text to its associated class.
#**********************************
#%%
# Import Libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report

#%%
# Loading Dataset
train_filename = r"Train.csv"
test_filename = r"Test_submission.csv"

#%%
#Q.C-1
# Create a df for train and test datasets

train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test_submission.csv")

#Q.C-2
# Conduct tokenization, Stop Word Removal, Lemmatization, and POS Tagging. 
# Write a function to wrap these steps together.

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def pos_tag_for_lemma(tag):
        return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}.get(tag[0], wordnet.NOUN)

    tokens = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in stop_words]

    return [(lemmatizer.lemmatize(w, pos_tag_for_lemma(t)), t) for w, t in nltk.pos_tag(tokens)]

#Q.C-3
# Apply the preprocessing function to the training and test data

train_df["clean_text"] = train_df["Text"].apply(preprocess_text)
test_df["clean_text"] = test_df["Text"].apply(preprocess_text)

#Q.C-4
# Select the classifier (you can choose 'nb' or 'lr')

train_df["clean_text_str"] = train_df["clean_text"].apply(lambda x: " ".join([w for w, t in x]))
test_df["clean_text_str"] = test_df["clean_text"].apply(lambda x: " ".join([w for w, t in x]))

X = train_df["clean_text_str"]
y = train_df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

#Q.C-5
# Conduct Text Classification, compare the performance of NB and Logistic reg classifiers

results = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))

    results[name] = y_pred

#Q.C-6. 
# Save the predictions to a CSV file (Please also submit the csv file to BB)

output_df = pd.DataFrame({
    "text": X_test,
    "NB_predicted_label": results["Naive Bayes"],
    "LR_predicted_label": results["Logistic Regression"]
})

output_df.to_csv("partC_predictions_comparison.csv", index=False)

#----------------------------------------The End--------------------------------------------









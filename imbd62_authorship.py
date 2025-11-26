# %% Load data
from datasets import load_dataset
import pandas as pd
import numpy as np


ds = load_dataset("tasksource/imdb62")
df = ds["train"].to_pandas()

# Rename and clean
df = df.rename(columns={"userId": "author", "content": "text"})
df["author"] = df["author"].astype(str)
df = df[["text", "author"]]

# Drop missing/empty reviews
df = df.dropna(subset=["text", "author"])
df = df[df["text"].str.strip() != ""]

df.to_csv("imdb62.csv", index=False)
print(df.head())

# %% Authorship Attribution 
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

import re, spacy

df = pd.read_csv("imdb62.csv")
print("Loaded:", df.shape, "authors:", df['author'].nunique())

# %% Feature helpers 
nlp = spacy.load("en_core_web_sm", disable=["ner","parser"])

def tokenize_simple(text):
    return re.findall(r"[A-Za-z']+|[0-9]+|[^\w\s]", text)

# punctuation style

def punctuation_style_stats(text):
    n = max(len(text), 1)

    commas = text.count(",") / n
    periods = text.count(".") / n
    excls = text.count("!") / n
    ques = text.count("?") / n
    quo = text.count("\"") / n
    apos = text.count("'") / n
    semi = text.count(";") / n
    colon = text.count(":") / n
    ddash = text.count("--") / n
    dash = (text.count("-") - 2 * text.count("--")) / n
    lpar = text.count("(") / n
    rpar = text.count(")") / n
    upper = sum(c.isupper() for c in text) / n

    return [
        commas, periods, excls, ques, quo, apos,
        semi, colon, dash, ddash, lpar, rpar, upper
    ]

# discourse markers

DISCOURSE_MARKERS = ["and", "but", "however", "though"]

def discourse_marker_rates(tokens):
    n = max(len(tokens), 1)
    lower = [t.lower() for t in tokens]
    return [lower.count(w) / n for w in DISCOURSE_MARKERS]


# vocabulary 

def vocab_richness(tokens):
    n = len(tokens)
    if n==0: return [0,0]
    types = len(set(tokens))
    ttr = types/n
    hapax = sum(1 for t in set(tokens) if tokens.count(t)==1)/n
    return [ttr, hapax]

# verb, adj, adv

def pos_richness(text):
    doc = nlp(text)

    groups = {
        "verb": {"VERB", "AUX"},
        "adj": {"ADJ"},
        "adv": {"ADV"},
    }

    feats = []
    for tags in groups.values():
        tokens = [
            tok.lemma_.lower()
            for tok in doc
            if tok.pos_ in tags and tok.is_alpha
        ]
        n = len(tokens)
        if n == 0:
            feats.extend([0.0, 0.0])
        else:
            types = set(tokens)
            ttr = len(types) / n
            hapax = sum(1 for w in types if tokens.count(w) == 1) / n
            feats.extend([ttr, hapax])

    return feats

# modals

MODALS = {
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "ought"
}

def modal_richness(text):
    doc = nlp(text)
    tokens = [tok.lemma_.lower() for tok in doc if tok.lemma_.lower() in MODALS]

    total_tokens = max(len(doc), 1)
    modal_rate = len(tokens) / total_tokens

    n = len(tokens)
    if n == 0:
        return [modal_rate, 0.0, 0.0]

    types = set(tokens)
    ttr = len(types) / n
    hapax = sum(1 for w in types if tokens.count(w) == 1) / n

    return [modal_rate, ttr, hapax]

# superlatives

def superlative_features(text):
    doc = nlp(text)
    total_tokens = max(len(doc), 1)

    most_count = sum(1 for tok in doc if tok.lower_ == "most")
    most_rate = most_count / total_tokens

    est_tokens = [
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha and tok.lower_.endswith("est")
    ]
    est_count = len(est_tokens)
    est_rate = est_count / total_tokens

    if est_count == 0:
        return [most_rate, est_rate, 0.0, 0.0]

    types = set(est_tokens)
    est_TTR = len(types) / est_count
    est_hapax = sum(1 for w in types if est_tokens.count(w) == 1) / est_count

    return [most_rate, est_rate, est_TTR, est_hapax]

# pronouns

FIRST_PERSON = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves"
}

SECOND_PERSON = {
    "you", "your", "yours", "yourself", "yourselves"
}

def pronoun_features(tokens):
    """
    Returns 2 features:
    [first_person_rate, second_person_rate]
    """
    n = max(len(tokens), 1)
    lower = [t.lower() for t in tokens]

    first = sum(1 for t in lower if t in FIRST_PERSON)
    second = sum(1 for t in lower if t in SECOND_PERSON)

    return [first / n, second / n]

# length

def length_digit_features(text, tokens):
    num_tokens = len(tokens)
    log_len = np.log(num_tokens + 1)
    total_chars = max(len(text), 1)
    digit_ratio = sum(c.isdigit() for c in text) / total_chars
    return [log_len, digit_ratio]


# %% Build features
def build_dense_features(texts):
    rows = []
    for t in texts:
        toks = tokenize_simple(t)

        feats = []
        feats.extend(discourse_marker_rates(toks))   # 4
        feats.extend(punctuation_style_stats(t))     # 13
        feats.extend(vocab_richness(toks))           # 2
        feats.extend(pos_richness(t))                # 6
        feats.extend(modal_richness(t))              # 3
        feats.extend(superlative_features(t))        # 4
        feats.extend(pronoun_features(toks))         # 2 
        feats.extend(length_digit_features(t, toks)) # 2 

        rows.append(np.array(feats, dtype=float).ravel())

    return np.vstack(rows)

# TF-IDF Blocks

def char_tfidf(train_texts, test_texts):
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        lowercase=True
    )
    return vec.fit_transform(train_texts), vec.transform(test_texts)

def word_tfidf(train_texts, test_texts):
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=5,
        max_features=50000,
        sublinear_tf=True
    )
    return vec.fit_transform(train_texts), vec.transform(test_texts)


# %%  Tain-eval

from sklearn.model_selection import StratifiedShuffleSplit

#  Single train/test split 
X = df["text"].tolist()
y = df["author"].astype(str).values

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(train_idx, test_idx) = next(splitter.split(X, y))

train_texts = [X[i] for i in train_idx]
test_texts  = [X[i] for i in test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

# --- sparse features ---
Xtr_char, Xte_char = char_tfidf(train_texts, test_texts)
Xtr_word, Xte_word = word_tfidf(train_texts, test_texts)

# --- dense features ---
Xtr_dense_raw = build_dense_features(train_texts)
Xte_dense_raw = build_dense_features(test_texts)

scaler = StandardScaler(with_mean=False)
Xtr_dense = scaler.fit_transform(Xtr_dense_raw)
Xte_dense = scaler.transform(Xte_dense_raw)

# Combine char + word + dense
Xtr = hstack([Xtr_char, Xtr_word, csr_matrix(Xtr_dense)], format="csr")
Xte = hstack([Xte_char, Xte_word, csr_matrix(Xte_dense)], format="csr")

# %% report
clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)
clf.fit(Xtr, y_train)
y_pred = clf.predict(Xte)

from sklearn.metrics import classification_report
acc = accuracy_score(y_test, y_pred)
mf1 = f1_score(y_test, y_pred, average="macro")
print(f"Accuracy={acc:.4f}  Macro-F1={mf1:.4f}")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))

# %% BiLSTM model 

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

print("\n=== BiLSTM experiment ===")

# 1. Tokenize text
max_words = 30000    # vocab size
max_len = 300        # max sequence length

tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
tokenizer.fit_on_texts(train_texts)

Xtr_seq = tokenizer.texts_to_sequences(train_texts)
Xte_seq = tokenizer.texts_to_sequences(test_texts)

Xtr_seq = pad_sequences(Xtr_seq, maxlen=max_len, padding="post", truncating="post")
Xte_seq = pad_sequences(Xte_seq, maxlen=max_len, padding="post", truncating="post")

# 2. Encode labels as integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

num_classes = len(le.classes_)
print("Num classes (authors):", num_classes)

# 3. Build BiLSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dense(num_classes, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# 4. Train
history = model.fit(
    Xtr_seq, y_train_enc,
    validation_split=0.1,
    epochs=3,          
    batch_size=32,
    verbose=1
)

# 5. Evaluate
pred_probs = model.predict(Xte_seq)
y_pred_enc = pred_probs.argmax(axis=1)

bilstm_acc = accuracy_score(y_test_enc, y_pred_enc)
bilstm_mf1 = f1_score(y_test_enc, y_pred_enc, average="macro")

print(f"\nBiLSTM Accuracy={bilstm_acc:.4f}  Macro-F1={bilstm_mf1:.4f}")



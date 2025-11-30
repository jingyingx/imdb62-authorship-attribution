# Feature engineering > Big Models: Authorship Attribution on IMDb62

This project shows that **smart feature engineering + linear models** can outperform both naïve neural architectures and transformer-based systems on stylometric authorship attribution. 

This repository implements a high-performing **non-transformer** baseline for authorship attribution on the IMDb62 dataset.  
The system combines:

- **Character TF–IDF** (3–5 char n-grams)  
- **Word TF–IDF** (1–2 word n-grams)  
- **Rich stylometric features**, including punctuation patterns, discourse markers, lexical richness, POS-based richness, modal usage, superlatives, pronouns, and length/digit metrics  
- A **LinearSVC** classifier  
- A **BiLSTM** sequence model as a neural baseline  

The project includes a complete pipeline: **data loading → preprocessing → feature engineering → training → evaluation** with accuracy and macro-F1.

---

# Motivation: Toward Beating BERTAA

Transformer-based authorship attribution systems such as **BERTAA** achieve strong results but often come with:

- high computational cost  
- limited interpretability  
- slower inference  
- risk of overfitting on stylistic data  

This project explores an alternative path:

> **How far can lightweight, interpretable, feature-rich models go—and can such a system ultimately compete with or surpass BERTAA?**

---

# Dataset: IMDB62

We use the IMDB62 dataset from HuggingFace:

- **62 authors**  
- **~1,000 reviews per author**  
- Text field: `content`  
- Author field: `userId`  

During preprocessing, the dataset is cleaned and saved as: data/imdb62.csv

---

# Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

This will install all Python packages required to run the feature engineering pipeline, LinearSVC classifier, and BiLSTM model.

---

# How to Run

Run the full pipeline:

```bash
python imdb62_authorship.py
```

This will:

- Download & preprocess IMDB62
- Extract TF–IDF and stylometric features
- Train a LinearSVC model
- Evaluate on a 20% test split
- Train a BiLSTM model
- Evaluate again

---

# Stylometric Features Used

- Discourse markers (and, but, however, though)
- Punctuation style (commas, periods, quotes, dashes…)
- Vocabulary richness (TTR, hapax)
- POS-based lexical richness (VERB, ADJ, ADV)
- Modal usage (can, will, would, may…)
- Superlatives (“most”, “*est”)
- Pronoun rates (1st and 2nd person)
- Length/digit statistics (log length, digit ratio)

---

# Models
1. LinearSVC (Primary Model)

Inputs:
- char TF–IDF
- word TF–IDF
- stylometric dense features

2. BiLSTM (Neural Baseline)

A simple neural text model:
- Keras tokenizer → integer sequences
- Embedding layer
- Bidirectional LSTM
- Softmax output over 62 authors
  
Not optimized; included as a deep learning comparison point.

---

# Results

Performance on an 80/20 stratified split (12,395 test reviews):

LinearSVC (TF–IDF + stylometric features)
-----------------------------------------
Accuracy:  0.9897
Macro-F1:  0.9897


BiLSTM (token-based sequence model)
-----------------------------------
Accuracy:  0.4252
Macro-F1:  0.4114

---

# Comparison with Published BERTAA Results

According to Fabien et al. (2020), the BERTAA + style + hybrid model
achieves ~93.0% accuracy on IMDB62 (Table 5, 10 training epochs).
Link: https://publications.idiap.ch/downloads/papers/2020/Fabien_ICON2020_2020.pdf

✔ Interpretation

The LinearSVC + stylometry pipeline (~99% accuracy) outperforms:
- a naive BiLSTM baseline
- the published BERTAA result (~93% accuracy) under comparable dataset conditions

This supports the finding that:

>**Smart feature engineering + linear models can outperform both naive neural models and transformer-based systems on stylometric authorship attribution tasks.**

---

# Author

Created by Jingying Xu

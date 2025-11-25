# Authorship Attribution on IMDB62 with TF–IDF, Stylometric Features, and Linear Models

This repository implements a high-performing **non-transformer** baseline for authorship attribution on the IMDB62 dataset.  
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

This repository establishes a **strong classical baseline** as the foundation for future comparisons against BERT-based models.

---

# Dataset: IMDB62

We use the IMDB62 dataset from HuggingFace:

- **62 authors**  
- **~1,000 reviews per author**  
- Text field: `content`  
- Author field: `userId`  

During preprocessing, the dataset is cleaned and saved as: data/imdb62.csv

# How to Run



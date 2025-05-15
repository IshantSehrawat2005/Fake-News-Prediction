# 📰 Fake News Detection using Machine Learning

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/NLP-Scikit--learn-orange)

## 📌 Project Overview

This project focuses on detecting **fake news articles** using Natural Language Processing (NLP) and supervised Machine Learning algorithms. The goal is to build a model that can classify whether a news article is **real** or **fake** based on its text content.

---

## 📂 Dataset

- **Source:** [Kaggle - Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Features:** `title`, `text`, `label`
- **Target column:** `label` (1 = Fake, 0 = Real)
- **Size:** ~44,000 articles

---

## ⚙️ Technologies Used

- 🐍 Python 3.8+
- 🧠 Scikit-learn
- 📖 NLTK
- ✨ Pandas, NumPy
- 📊 Matplotlib, Seaborn
- 🧪 TF-IDF Vectorizer
- 💻 Jupyter Notebook

---

## 🧠 ML Algorithms Implemented

- Logistic Regression
- Passive Aggressive Classifier
- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)

---

## 🔍 Key Steps

1. **Data Preprocessing**
   - Removed nulls and duplicates
   - Lowercased text
   - Removed punctuation, stopwords, and URLs
   - Tokenized and cleaned using NLTK

2. **Feature Extraction**
   - TF-IDF Vectorization (Text to Numeric Matrix)
   - Explored CountVectorizer (optional)

3. **Model Building**
   - Trained multiple classifiers
   - Hyperparameter tuning with GridSearchCV
   - Evaluated model performance using multiple metrics

4. **Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC Curve

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**

> 🎯 Focus on high precision and recall to minimize both false positives and false negatives in fake news detection.



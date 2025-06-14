# Logistic Regression on Fake vs Real News Dataset


# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load datasets
fake_df = pd.read_csv("/fake-and-real-news-dataset/Fake.csv")
real_df = pd.read_csv("/fake-and-real-news-dataset/True.csv")

# 3. Add labels
fake_df['label'] = 1  # Fake news
real_df['label'] = 0  # Real news

# 4. Combine datasets
data = pd.concat([fake_df, real_df], ignore_index=True)

# 5. Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Keep only text and label (you can add title later if needed)
data = data[['text', 'label']]

# 7. Basic text preprocessing (optional starter)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # remove brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove links
    text = re.sub(r'<.*?>+', '', text)  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)  # remove line breaks
    text = re.sub(r'\w*\d\w*', '', text)  # remove words with numbers
    return text

data['text'] = data['text'].apply(clean_text)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)

# 9. Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 10. Train a model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# 11. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()



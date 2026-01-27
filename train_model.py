import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess dataset
df = pd.read_csv('fake reviews dataset.csv')
df.dropna(inplace=True)

# Text preprocessing function
def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Split data into training and testing sets
review_train, review_test, label_train, label_test = train_test_split(df['text_'], df['label'], test_size=0.35, random_state=42)

# Logistic Regression Pipeline
pipeline_lr = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the model
pipeline_lr.fit(review_train, label_train)

# Save the model
joblib.dump(pipeline_lr, 'model.pkl')
print("Model saved as model.pkl")

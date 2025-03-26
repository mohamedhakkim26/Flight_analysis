import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('problem2/api_keys.env')

file_path = os.path.join('problem2', 'dataset.csv')

# Function to load and clean data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['Review', 'Recommended']].dropna()
    data['Label'] = data['Recommended'].map({'yes': 'positive', 'no': 'negative'})
    return data[['Review', 'Label']]

# Function to train model
def train_sentiment_model(training_data):
    X = training_data['Review']
    y = training_data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    model_pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = model_pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join('problem2', 'sentiment_model.pkl')
    joblib.dump(model_pipeline, model_path)

    return model_pipeline

# Function to predict sentiment
def predict_sentiment(model, new_text):
    prediction = model.predict([new_text])[0]
    return prediction

if __name__ == "__main__":
    dataset_path = os.path.join('problem2', 'AirlineReviews.csv')
    data = load_data(dataset_path)

    model = train_sentiment_model(data)

    # Example Predictions
    print(predict_sentiment(model, "The flight was fantastic!"))
    print(predict_sentiment(model, "Horrible experience, never flying again!"))

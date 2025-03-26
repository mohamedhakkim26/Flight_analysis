# README.md

## Airline Sentiment Analysis
This project implements a binary sentiment classification model for airline customer feedback.

### Project Structure

problem2/
├── main.py            # Model training and prediction logic
├── api_keys.env       # Environment variables for API keys
├── requirements.txt   # Required libraries
├── README.md          # Instructions and documentation

### Installation
1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set API Keys:**
Create a `.env` file or use `api_keys.env` to store your keys securely.

### Usage
1. **Train the model:**
Run the following command to train the sentiment model:
```bash
python main.py
```

2. **Example Predictions:**
```python
from main import predict_sentiment, train_sentiment_model

dataset_path = 'problem2/AirlineReviews.csv'
model = train_sentiment_model(load_data(dataset_path))

print(predict_sentiment(model, "The flight was fantastic!"))
print(predict_sentiment(model, "Horrible experience, never flying again!"))
```

### Dataset Structure
The dataset should be a CSV file with the following columns:
- **Review**: Text of the customer review
- **Recommended**: 'yes' for positive sentiment, 'no' for negative sentiment

### Additional Notes
- The model uses a **TF-IDF vectorizer** combined with **Logistic Regression**.
- Evaluation metrics include **accuracy**, **confusion matrix**, and **classification report**.

### Future Improvements
- Implement more advanced models like **Naive Bayes**, **Random Forest**, or **Deep Learning**.
- Add improved text cleaning techniques for better accuracy.


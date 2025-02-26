from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from langdetect import detect, LangDetectException
from sklearn.preprocessing import LabelEncoder
from googleapiclient.discovery import build
from sklearn.metrics import accuracy_score
import pandas as pd
import re


def load_and_clean_data():
    df = pd.read_csv('Reddit_Data.csv')
    # df = pd.read_csv('cleaned_df.csv')
    df = df.dropna()

    def clean_comment(comment):
        comment = re.sub(r'<.*?>', '', comment)
        comment = re.sub(r'[^A-Za-z0-9\s.,!?\'"]+', '', comment)
        comment = re.sub(r'\s+', ' ', comment).strip()
        return comment

    df['clean_comment'] = df['clean_comment'].apply(clean_comment)
    return df

df = load_and_clean_data()

label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_comment'])
y = df['category_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

API_KEY = 'AIzaSyDxLMRyAjAUlVljXFl5WNBPxxImsZgDBdw'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def fetch_comments(video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100)
        response = request.execute()
        while request:
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(part='snippet', videoId=video_id,pageToken=response['nextPageToken'], maxResults=100)
                response = request.execute()
            else:
                break
    except Exception as e:
        print(f"Error fetching comments: {e}")
    return comments

def preprocess_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(user_input):
    user_input_cleaned = preprocess_text(user_input)
    user_input_vectorized = tfidf.transform([user_input_cleaned])
    prediction = model.predict(user_input_vectorized)
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_mapping.get(prediction[0], 'Unknown')

def clean_comment(comment):
    comment = re.sub(r'<.*?>', '', comment)
    comment = re.sub(r'[^a-zA-Z\s]', '', comment)
    comment = comment.lower()
    comment = re.sub(r'\s+', ' ', comment).strip()
    return comment

def analyze_video_comments(video_id):
    comments = fetch_comments(video_id)
    positive_comments, neutral_comments, negative_comments = [], [], []
    for comment in comments:
        try:
            if detect(comment) != 'unknown':
                cleaned_comment = clean_comment(comment)
                if len(cleaned_comment.split()) > 70:
                    continue
                sentiment = predict_sentiment(cleaned_comment)
                if sentiment == 'Positive':
                    positive_comments.append(cleaned_comment)
                elif sentiment == 'Neutral':
                    neutral_comments.append(cleaned_comment)
                elif sentiment == 'Negative':
                    negative_comments.append(cleaned_comment)
        except LangDetectException:
            continue
    return positive_comments, neutral_comments, negative_comments
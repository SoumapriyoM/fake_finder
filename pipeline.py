import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

class FakeNewsPipeline:
    def __init__(self, model_path='model.pkl', tfidf_path='tfidf_vectorizer.pkl'):
        self.loaded_model = pickle.load(open(model_path, 'rb'))
        self.tfidf_vectorizer = pickle.load(open(tfidf_path, 'rb'))

    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and non-word characters
        words = nltk.word_tokenize(text.lower())  # Tokenization and lowercase conversion
        stop_words = set(stopwords.words('english'))
        stop_words.discard('no')  # Remove 'no' from the set of stopwords
        stop_words.discard('not')  # Remove 'not' from the set of stopwords
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Stopwords removal and lemmatization
        return ' '.join(words)

    def predict_fake_news(self, news):
        input_data = self.preprocess_text(news)
        input_data = [input_data]
        vectorized_input_data = self.tfidf_vectorizer.transform(input_data)
        prediction = self.loaded_model.predict(vectorized_input_data)[0]
        return prediction

# dataframe = pd.read_csv('New_train.csv')
# ch = pd.read_csv("test.csv")
# x = dataframe['news']
# y = dataframe['label']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
# tfid_x_train = tfidf_vectorizer.fit_transform(x_train)
# tfid_x_test = tfidf_vectorizer.transform(x_test)

# with open('tfidf_vectorizer.pkl', 'wb') as tfidf_file:
#     pickle.dump(tfidf_vectorizer, tfidf_file)

# pipeline = FakeNewsPipeline(model_path='model.pkl', tfidf_path='tfidf_vectorizer.pkl')

# # Example news text for testing
# test_news_text = "This is a test news article."

# # Make a prediction using the pipeline
# prediction_result = pipeline.predict_fake_news(test_news_text)

# # Get and print the result
# result_message = "FAKE News" if prediction_result == 0 else "REAL News"
# print(f"Prediction: {result_message}")


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json

nltk.download('stopwords')
nltk.download(['stopwords', 'wordnet'])
nltk.download('punkt')

stop_words = stopwords.words('english')
stop_words = set(stop_words)
lemmatizer = WordNetLemmatizer()

def clean_text(cleaned_text):
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove multiple spaces
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_text)  # Remove special characters
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.strip()
    tokens = nltk.word_tokenize(cleaned_text)
    cleaned_text = [word for word in tokens if word not in stop_words]
    cleaned_text = [lemmatizer.lemmatize(word) for word in cleaned_text]
    cleaned_text = ' '.join(cleaned_text)
    return cleaned_text

def json_to_str(row):
    keywords = json.loads(row)
    names = ', '.join([kw['name'].replace(" ", "").replace("'", "").replace('"', '') for kw in keywords])
    return names

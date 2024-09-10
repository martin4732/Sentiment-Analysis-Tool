from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    tagged_tokens = pos_tag(tokens)
    filtered_tokens = [word for word, pos in tagged_tokens if pos in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']]
    return ' '.join(filtered_tokens)

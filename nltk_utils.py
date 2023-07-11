import nltk
from nltk.stem.porter import PorterStemmer

# need to download this once which is a pre-trained model for sentence tokenization
# nltk.download('punkt')

stemmer = PorterStemmer()

def token(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = [1 if word in tokenized_sentence else 0 for word in all_words]
    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["whats", "up", "shawty", "hello", "how", "doing", "are", "you", "faring", "good"]
# print(bag_of_words(sentence, words))

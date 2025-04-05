from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

text = "This is a test sentence for NLTK tokenization."
words = word_tokenize(text)

print(words)

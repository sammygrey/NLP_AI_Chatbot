import nltk 

# Bottom ssl is workaround for broken script on punkt donwloadm which returns a loading ssl error
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
#End of error workaround

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#Imports needed from nltk

#Our Tokenizer
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#Stemming Function
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

#Test of our imports, Tokenizer, and Stemming.
#Tokenizer
test = "Is anyone there?"
print("Before tokenization")
print(test)
print("_____________")
print("After Tokenization")
test = tokenize(test)
print(test)
#What we should expect from above is our original sentence of "Is Anyone There?" 
#After undergoing tokenization, we should see our sentence split up
#Use python nltk_utils.py to run

#Stemming
words = ["Organize", "organizes", "organizing"]
print("Before Stemming")
print(words)
print("_________________")
stemmed_words = [stem(w) for w in words]
print("After Stemming")
print(stemmed_words)

#We see our sentence is broken down to its root word of organ, organ, and organ with suffix and prefixes


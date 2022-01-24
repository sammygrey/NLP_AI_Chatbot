import nltk 
import numpy as np
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
    #When we use this function, we have our tokenized sentence as well as all our words
    #So we look at each word in the sentence, and if it is available in the all_words array, 
    #We put a 1 at the position of where that word is found
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    #Now we create our "bag" and initialize with all 0s

    bag = np.zeros(len(all_words), dtype=np.float32)
    #Now we loop for our all words, so for index and words in all our words
    # We check, if this word is in our tokenized sentence, it will return a value of 1 at the specified index
    # Then we will return the bag
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag

#With the below print statement, we see the function of Bag of Words in operatio
print("Testing our bag_of_words function")
sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bog = bag_of_words(sentence, words)
print(bog)
print("--------------")

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
#Use python nltk_utils.py to run, if using python version 3.X.X, using python3 nltk_utils.py to run

#Stemming
words = ["Organize", "organizes", "organizing"]
print("Before Stemming")
print(words)
print("_________________")
stemmed_words = [stem(w) for w in words]
print("After Stemming")
print(stemmed_words)

#We see our sentence is broken down to its root word of organ, organ, and organ with suffix and prefixes


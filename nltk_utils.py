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

#Bag of Words Function
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


#Test our function with the below sentence to visualize Tokenization. 

#What is the purpose of tokenizing our text?:
#Tokenizing our text allows us to use the individual parts of it, be that words, phrases, or punctuation!
#Testing our Tokenizer
test_sentence = "I will not live in peace until I find the Avatar!"
tokens = tokenize(test_sentence)
for term in tokens:
        print(term)

#Test our Stemming function on the below words. 

#How does stemming affect our data?:
#Stemming is the process of removing part of a word, in our case we are trying to 
#get the root of the word so we will chop off plural indicators, past tense, stuff like that
words = ["Organize", "organizes", "organizing", "disorganized"]
for word in words:
    print(stem(word))

#Implement the above Bag of Words function on the below sentence and words. 

#What does the Bag of Words model do? Why would we use Bag of Words here instead of TF-IDF or Word2Vec?:
#The bag of words model converts the words/tokens we have provided to a pattern of numbers that the neural net can understand. 
#Words in a sentence are assigned avalue of 1 while non-present words are assigned 0.
#We wouldn't want to use tf-idf or word2vec here because we don't really have a preexisting corpus to quantify the weight of terms 
#in a document, it makes more sense to pass in the sentence and words into a bag of words model in this case

print("Testing our bag_of_words function")
sentence = ["I", "will", "now", "live", "in", "peace", "until", "I", "find", "the", "Avatar"]
words = ["hi", "hello", "I", "you", "the", "bye", "in", "cool", "wild", "find"]
print(bag_of_words(sentence,words))
print("--------------")



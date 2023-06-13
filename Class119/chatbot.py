import nltk
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np

words = []
classes = []
word_tag_list = []
ignore_words = ["?","!",".",",","'s"]
train_data_file = open("intents.json").read()
intents = json.loads(train_data_file)

stemmer = PorterStemmer()

def findStemWords(words,ignore_words):
    stem_words=[]
    for i in words:
        if i not in  ignore_words:
            stemWord = stemmer.stem(i.lower())
            stem_words.append(stemWord)
    return(stem_words)        
        
#chatbot corpus <--- expected input from the user
for i in intents["intents"]:
    for a in i["patterns"]:
        patternWords = nltk.word_tokenize(a)
        #extend() <-- adds elements from an list one by one to another list
        words.extend(patternWords)
        word_tag_list.append((patternWords, i["tag"] ))
    if i["tag"] not in classes:
        classes.append(i["tag"])
        stem_words = findStemWords(words,ignore_words) 

# print(word_tag_list[0])
# print(classes)
# print(stem_words)

def chatBotCorpus(stem_words,classes):
    stemWords = sorted(list(set(stem_words))) 
    classes = sorted(list(set(classes))) 
    # sorted is inbuilt function that sorts "stem_words" into least to greatest with a copy (see 2)
    # (2) it does not modify the original list instead ot will return a new sorted list
    # sorted() is used in tuples, list and for strings
    # sort() - it will get applied directly to the list
    # pickle converts an object into byte string (good for transferring data)
    pickle.dump(stemWords, open("words.pkl","wb"))
    pickle.dump(classes, open("classes.pkl","wb"))
    return stemWords, classes
    
stem_words, classes = chatBotCorpus(stem_words,classes)

# print(stem_words)
# print(classes) # classes are the tags

training_data = []
number_of_tags = len(classes)
labels = [0]*number_of_tags

for i in word_tag_list:
    bag_of_words = []
    pattern_words = i[0]

    for word in pattern_words:
        index = pattern_words.index(word)
        word = stemmer.stem(word.lower())
        pattern_words[index] = word
    
    for word in stem_words:
        if word in pattern_words:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    # print(bag_of_words)
    
    label = list(labels) #entire element will be zero
    tag = i[1] 
    tag_index = classes.index(tag)
    label[tag_index] = 1
    training_data.append([bag_of_words,label])
# print(training_data[0])

def preprocess_Training(training_data):
    training_data = np.array(training_data,dtype=object)
    train_x = list(training_data[:,0])
    train_y = list(training_data[:,1])
    print(train_x[0])
    print(train_y[0])
    return(train_x, train_y)
    
train_x , train_y = preprocess_Training(training_data)









    
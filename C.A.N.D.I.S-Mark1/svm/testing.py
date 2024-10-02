import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
# from nltk.corpus.reader import averaged_perceptron_tagger
import time
from time import strftime
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os
np.random.seed(500)
'''
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
#TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
'''

Corpus = pd.read_csv(r"ISEAR.csv",encoding='utf-8')

# Step - 1: Data Pre-processing - This will help in getting better results through the classification algorithms

# Step - 1a : Remove blank rows if any.
Corpus['Text'].dropna(inplace=True)

# Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['Text'] = [entry.lower() for entry in Corpus['Text']]

# Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['Text']= [word_tokenize(entry) for entry in Corpus['Text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(Corpus['Text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)

#print(Corpus['text_final'].head())

# Step - 2: Split the model into Train and Test Data set
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Emotion'],test_size=0.2)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

print(Train_Y)
print(Test_Y)

print(Corpus['text_final'])

sentences=Corpus['text_final']

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer(ngram_range=(1,1))
x_train_dtm=count_vect.fit_transform(Train_X)
x_test_dtm=count_vect.transform(Test_X)
#
SVM = svm.SVC(C=1.0, kernel='linear', degree=8, gamma='auto')
SVM.fit(x_train_dtm,Train_Y)


SVM2=LinearSVC()
SVM2.fit(x_train_dtm,Train_Y)

predictions_SVM = SVM.predict(x_test_dtm)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

predictions_SVM2 = SVM2.predict(x_test_dtm)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM2, Test_Y)*100)
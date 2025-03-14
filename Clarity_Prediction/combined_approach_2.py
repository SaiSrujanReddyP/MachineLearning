# this is the stutter approach for identifying the clarity of what is spoken by the student
# we vectorize the transcribed text based on the repetition of words. If a worda repeats right after each other, it is called a 0th order stutter
# if a word repeats leaving 1 word gap, it is called a first order stutter and so on, if a word leaves gap of n words, before repeating, it is called an nth order stutter.
# The model predicts the clarity based on the number of each of these stutters
# the hyperparameters in this method consists of the ML model used and order of stutter being considered.
# multiple methods were compared to find out the best ML model for stutters uptil 3rd order. 
# MLP classifier was chosen for the classification based on this method.
# Then the next the order of words to be chosen is chosen by backward reduction techniques.
# It was observed that uptil 3rd order of stutters probided best accuracy

import random
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import copy
import pandas as pd
from collections import Counter
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB  
import matplotlib.pyplot as plt

dataset = 'vivaData.csv'

def roundoff(number):
  if number >= 0.5:
    return 1
  else:
    return 0

def checkAccuracy(clf, y_test, X_test):
  y_pred = clf.predict(X_test)
  try:
    confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    confusionMatrixDisplay.plot()
    plt.show()
  except ValueError:
    print(f"ERROR:------------------------------------------\n{y_pred}\n{y_test}")
  # check scores

  try:
    F1_score = f1_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print({"Precision":Precision,"recall":recall,"F1_score":F1_score})
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
  except ValueError:
    print("error......")
  
def checkAccuracy2(y_test, layer1:list, layer2):
  y_pred = []
  for i in range(len(layer1)):
    if(layer1[i] == 0):
      y_pred.append(layer1[i])
    else:
      y_pred.append(layer2[i])
  # F1_score = f1_score(y_test, y_pred)
  
  confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
  confusionMatrixDisplay.plot()
  plt.show()
  # check scores
  print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

df = pd.read_csv(dataset,encoding='windows-1252')
# df = df.tail(429 - 161)
wc = Counter()
for answer in df["answer"]:
  answer = str(answer)
  answer = answer.lower()
  wc.update(Counter(word_tokenize(answer)))

# print(wc)


wc = Counter()
for answer in df["answer"]:
  answer = str(answer)
  answer = answer.lower()
  wc.update(Counter(word_tokenize(answer)))

# print(wc)

# we only consider the most common words out of all
com_words = dict(wc.most_common(40))

vocabulary_count = dict(wc)
vocabulary = list(wc.keys())

# now take each of these words one by one, in order to predict for clarity.
# we use term frequency vectorizer for these 40 words
# instead of doing vectorization, another approach would be to remove each of the words in order and see the new length of the answer
# we classify clarity based on the ratio between the previous length of the answer and the new length.

tf_vectors = []
for answer in df['answer']:
  answer = str(answer)
  answer = answer.lower()
  term_freq = []
  for word in com_words.keys():
    term_freq.append(answer.count(word))
  # print(term_freq)
  tf_vectors.append(term_freq)

max_stutter_order = 3
stutters = []
for answer in df['answer']:
  answer_words = word_tokenize(str(answer).lower())
  # the 3 commonly identified stutters
  isdbst = 0  # variable to check whether it is a double word stutter.
  istpst = 0  # to check whether it is a triple word stutter.

  double_word_stutters = 0
  triple_word_stutters = 0

  stutter = [0 for i in range(max_stutter_order)]

  for i in range(len(answer_words) - 1):
    isdbst = 0
    for j in range(max_stutter_order):
      if i < len(answer_words) - j - 1 and answer_words[i] == answer_words[i + j + 1]:
        stutter[j] += 1
        isdbst += 1
        istpst += 1
    if isdbst == 2:
      double_word_stutters += 1
    if istpst == 3:
      triple_word_stutters += 1
  
  stutter.append(double_word_stutters)
  stutter.append(triple_word_stutters)
  stutters.append(stutter)

###################################################################################
#################### CLASSIFICATION USING ML ######################################
###################################################################################
# now, out of these, we use dimensionality reduction to choose the most useful words
# decision tree classifier is used.


# popouts = [28,13,8,7,6,4,2,1] # added extra based on observation => 4,7,
# popouts = [7]
# for i in popouts:
#   for j in range(len(tf_vectors)):
#     tf_vectors[j].pop(i)

###################################################################################
#################### CLASSIFICATION USING ML ######################################
###################################################################################
# now, out of these, we use dimensionality reduction to choose the most useful words
# decision tree classifier is used.


randomState = random.randint(0,100)
# randomState = 20
print("random state ->", randomState)
tf_vectors.extend(stutters)
X = tf_vectors
y = df['clarity'].values.astype('b')

count_1 = 0
count_0 = 0
for val in y:
  if val == 1:
    count_1 += 1
  elif val == 0:
    count_0 += 1
print("0:",count_0,";1:",count_1)


smote=SMOTE(sampling_strategy='minority') 
X,y=smote.fit_resample(X,y)
count_1 = 0
count_0 = 0
for val in y:
  if val == 1:
    count_1 += 1
  elif val == 0:
    count_0 += 1
print("0:",count_0,";1:",count_1)

# print('random state ->', randomState)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=randomState)

clf = MLPClassifier(max_iter=3000)
clf.fit(X_train,y_train)

checkAccuracy(clf,y_test,X_test)

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

clf = RandomForestClassifier()
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
      else:
        isdbst -= 1
        istpst -= 1
      
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


randomState = random.randint(0,100)
# randomState = 32
print('random state ->', randomState)

X = stutters
y = df['clarity'].values.astype('b')
# smote=SMOTE(sampling_strategy='minority') 


# X,y=smote.fit_resample(X,y)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=randomState)
  # X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.99, random_state=randomState)

# clf = MLPClassifier(max_iter=3000)
clf = svm.SVC()

X_test = X_temp
y_test = y_temp
clf.fit(X_train,y_train)
checkAccuracy(clf,y_test,X_test)

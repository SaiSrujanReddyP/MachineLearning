# this file is for the combined approach of using the term frequency method in layer 2 with a stacking classifier and stutter method in layer 1 using MLP classifier
# this approach is based on the fact that the mlp classifier provides excellent recall hence it is very unlikely to give false negatives.
# hence the mlp classifier is initially used to weed out the negatives and then layer 2 will again classify what is positive as positive or negative.

# result is, adaboost classifier was then used for finding out the accuracy of this approach.
#  it is able to achieve an accuracy of 77%



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


#########################################################################################
#################### PREDEFINED METHODS #################################################
#########################################################################################
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
    print("Precision",Precision,"recall",recall,"F1_score",F1_score)
    print('Accuracy',accuracy_score(y_test, y_pred))
  except ValueError:
    print("error......")
  
def checkAccuracy2(y_test, layer1, layer2):
  y_pred = []
  for i in range(len(layer1)):
    if(layer1[i] == 0):
      y_pred.append(layer1[i])
    else:
      y_pred.append(layer2[i])
  
  confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
  confusionMatrixDisplay.plot()
  plt.show()
  # check scores
  F1_score = f1_score(y_test, y_pred)
  Precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  print("Precision",Precision,"recall",recall,"F1_score",F1_score)
  print('Accuracy',accuracy_score(y_test, y_pred))




df = pd.read_csv(dataset,encoding='windows-1252')
wc = Counter()
for answer in df["answer"]:
  answer = str(answer)
  answer = answer.lower()
  wc.update(Counter(word_tokenize(answer)))



wc = Counter()
for answer in df["answer"]:
  answer = str(answer)
  answer = answer.lower()
  wc.update(Counter(word_tokenize(answer)))

# we only consider the most common words out of all
com_words = dict(wc.most_common(34))

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
  tf_vectors.append(term_freq)

max_stutter_order = 3  # max order considered for alternate word stutters.
stutters = []
for answer in df['answer']:
  answer_words = word_tokenize(str(answer).lower())
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

randomState = random.randint(0,100)
print('random state ->', randomState)

X = stutters
y = df['clarity'].values.astype('b')

# SMOTE is not used for class imbalance as the synthesized dataset is not uniform for stutter and common words.

X_train, X_test_1, y_train, y_test_1 = train_test_split(X, y, test_size=0.3, random_state=randomState)

clf = MLPClassifier(max_iter=4000)
clf.fit(X_train,y_train)

# prediction given by layer 1 relates to the stutters.
layer_1_predictions = clf.predict(X_test_1)

##############################################################################################------------------


X = tf_vectors


clf1 = MLPClassifier(max_iter=4000)
clf2 = RandomForestClassifier()

print(clf1.get_params())
print(clf2.get_params())


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=randomState)
# the data is split into test data as well but it will remain unutilised so that there are gaps in the data which is used for training the model.
X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=randomState)

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
m1 = clf1.predict(X_temp)
m2 = clf2.predict(X_temp)

meta_train = []
for i in range(len(m1)):
  meta_train.append([m1[i],m2[i]])

clf_meta = svm.SVC()
clf_meta.fit(meta_train,y_temp)

######################## CHECKING ACCURACY OF META MODEL ###########################

X_train, X_test_2, y_train, y_test_2 = train_test_split(X, y, test_size=0.3, random_state=randomState)


m1_ = clf1.predict(X_test_2)
m2_ = clf2.predict(X_test_2)
meta_test = []
for i in range(len(m1_)):
  meta_test.append([m1_[i],m2_[i]])
y_pred = clf_meta.predict(meta_test)
try:
  confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test_2, y_pred))
  confusionMatrixDisplay.plot()
  plt.show()
except ValueError:
  print(f"ERROR:------------------------------------------\n{len(y_pred)}\n{len(y_test_2)}")
# check scores

try:
  F1_score = f1_score(y_test_2, y_pred)
  Precision = precision_score(y_test_2, y_pred)
  recall = recall_score(y_test_2, y_pred)
  print({"Precision":Precision,"recall":recall,"F1_score":F1_score})
  print(f'Accuracy: {accuracy_score(y_test_2, y_pred)}')
except ValueError:
  print("error......")

y_pred_temp = clf1.predict(X_test_2)
y_pred_layer1 = [roundoff(pr) for pr in y_pred_temp]
print("prediction -> ",y_pred_layer1)
########################################################################################################-------------------

# prediction given by layer 2 is from the stacking model
layer_2_predictions = y_pred

checkAccuracy2(y_test_2,layer_1_predictions,layer_2_predictions)


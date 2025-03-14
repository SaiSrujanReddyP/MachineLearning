# clarity classification with stacking of MLP classifier and random forest classifier with use of desicion tree to give final output
import random
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import copy
import pandas as pd
from collections import Counter
from nltk import word_tokenize
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB  
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

dataset = 'vivadata.csv'

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

###################################################################################
#################### CLASSIFICATION USING ML ######################################
###################################################################################
# now, out of these, we use dimensionality reduction to choose the most useful words
# decision tree classifier is used.


randomState = random.randint(0,100)
print("random state -> ",randomState)
randomState = 23
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


##################################################################################
######################### STACKING MODELS ########################################
##################################################################################



clf1 = MLPClassifier(max_iter=4000)
clf2 = RandomForestClassifier()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=randomState)
X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=randomState)

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
m1_ = clf1.predict(X_test)
m2_ = clf2.predict(X_test)
meta_test = []
for i in range(len(m1_)):
  meta_test.append([m1_[i],m2_[i]])
y_pred = clf_meta.predict(meta_test)
try:
  confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
  confusionMatrixDisplay.plot()
  plt.show()
except ValueError:
  print(f"ERROR:------------------------------------------\n{len(y_pred)}\n{len(y_test)}")
# check scores

try:
  F1_score = f1_score(y_test, y_pred)
  Precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  print({"Precision":Precision,"recall":recall,"F1_score":F1_score})
  print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
except ValueError:
  print("error......")

y_pred_temp = clf1.predict(X_test)
y_pred_layer1 = [roundoff(pr) for pr in y_pred_temp]
print("prediction -> ",y_pred_layer1)
# accuracy of first layer checking

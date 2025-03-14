import random
from sklearn.ensemble import AdaBoostClassifier
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
  # print(term_freq)
  tf_vectors.append(term_freq)

###################################################################################
################### Identification of Stuttering ##################################
###################################################################################
max_stutter_order = 3
stutters = []
for answer in df['answer']:
  answer_words = word_tokenize(str(answer).lower())
  # the 3 commonly identified stutters
  single_word_stutters = 0
  alternate_single_stutter = 0
  second_order_stutter = 0
  third_order_stutter = 0
  stutter = [0 for i in range(max_stutter_order)]

  for i in range(len(answer_words) - 1):
    for j in range(max_stutter_order):
      if i < len(answer_words) - j - 1 and answer_words[i] == answer_words[i + j + 1]:
        stutter[j] += 1
  stutters.append(stutter)
  
# print("##################### stutters ########################################")
# print(stutters)
# for i in df:
#   print(i)

###################################################################################
#################### CLASSIFICATION USING ML ######################################
###################################################################################
# now, out of these, we use dimensionality reduction to choose the most useful words
# decision tree classifier is used.


randomState = random.randint(0,100)
print('random state ->', randomState)
# appending stutter counts to the word frequency counts

clf1 = DecisionTreeClassifier()
clf2 = GaussianNB()
clf3 = svm.SVC()
clf4 = KNeighborsClassifier()
clf5 = AdaBoostClassifier()
clf6 = RandomForestClassifier()

X_train, X_temp, y_train, y_temp = train_test_split(stutters, df['clarity'].values.astype('b'), test_size=0.6, random_state=randomState)
X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=randomState)

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
clf4.fit(X_train,y_train)
clf5.fit(X_train,y_train)
clf6.fit(X_train,y_train)

m1 = clf1.predict(X_temp)
m2 = clf2.predict(X_temp)
m3 = clf3.predict(X_temp)
m4 = clf4.predict(X_temp)
m5 = clf5.predict(X_temp)
m6 = clf6.predict(X_temp)



meta_train = []
for i in range(len(m1)):
  meta_train.append([m1[i],m2[i],m3[i],m4[i],m5[i],m6[i]])

clf_meta = MLPClassifier(max_iter=5000)
clf_meta.fit(meta_train,y_temp)

######################## CHECKING ACCURACY OF META MODEL ###########################
m1_ = clf1.predict(X_test)
m2_ = clf2.predict(X_test)
m3_ = clf3.predict(X_test)
m4_ = clf4.predict(X_test)
m5_ = clf4.predict(X_test)
m6_ = clf4.predict(X_test)
meta_test = []
for i in range(len(m1_)):
  meta_test.append([m1_[i],m2_[i],m3_[i],m4_[i],m5_[i],m6_[i]])
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

checkAccuracy(clf1,y_test, X_test)
checkAccuracy(clf2,y_test, X_test)
checkAccuracy(clf3,y_test, X_test)
checkAccuracy(clf4,y_test, X_test)
checkAccuracy(clf5,y_test, X_test)
checkAccuracy(clf6,y_test, X_test)




# removing features reduced based on backward selection
# for i in range(len(tf_vectors)):
#   tf_vectors[i].pop(6)

clf2 = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(tf_vectors, df['clarity'].values.astype('b'), test_size=0.18, random_state=randomState)
clf2.fit(X_train,y_train)

# accuracy of 2nd layer checking
y_pred_layer2 = clf2.predict(X_test)
checkAccuracy(clf2,y_test, X_test)

# checking accuracy
checkAccuracy2(y_test,y_pred_layer1,y_pred_layer2)


exit()


##################################################################################
################## DIMENSIONALITY REDUCTION ######################################
##################################################################################

y_pred = clf.predict(X_test)
F1_score = f1_score(y_test, y_pred)
max_acc = F1_score
print(max_acc)

ogtf = copy.deepcopy(tf_vectors)
print(f"ogtf contains {len(ogtf)} rows and {len(ogtf[0])} columns")

acc_increasing = True
# commencing backward selection for dimensionality reduction
while(acc_increasing):
  popout = -1
  acc_increasing = False
  for i in range(len(ogtf[0])): # len(ogtf[0]) contains the number of columns
    tf_vectors = copy.deepcopy(ogtf)
    try:
      for j in range(len(tf_vectors)):
        #print(j,str(len(tf_vectors[0])))
        tf_vectors[j].pop(i)
    except IndexError:
      
      print("trying to pop out " + str(i) + "while size of vector is " + str(len(tf_vectors[0])))
      exit()
    # now, out of these, we use dimensionality reduction to choose the most useful words
    X_train, X_test, y_train, y_test = train_test_split(tf_vectors, df['clarity'].values.astype('b'), test_size=0.2, random_state=0)
    clf = KNeighborsClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    F1_score = f1_score(y_test, y_pred)
    if(F1_score > max_acc):
      popout = i
      acc_increasing = True
      max_acc = F1_score

  if (popout > -1):
    # popout particular column from all the rows
    print(f"popping out {popout}")
    for i in range(len(ogtf)):
      ogtf[i].pop(popout)

print(max_acc)
print(f"ogtf contains {len(ogtf)} rows and {len(ogtf[0])} columns")

###################################################################################
###################################################################################
###################################################################################
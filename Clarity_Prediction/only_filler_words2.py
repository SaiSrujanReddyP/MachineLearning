# mlp classifier and random forest models are hyper-parmameter tuned and backward dimensionality reduction is performed in order to maximize prediction accuracy.
import random
from sklearn.model_selection import GridSearchCV
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
  
  confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
  confusionMatrixDisplay.plot()
  plt.show()
  # check scores
  print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

df = pd.read_csv(dataset,encoding='windows-1252')

wc = Counter()
for answer in df["answer"]:
  answer = str(answer)
  answer = answer.lower()
  wc.update(Counter(word_tokenize(answer)))

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
  tf_vectors.append(term_freq)


# initially, the previously popped words were removed directly with the assumption that the accuracy will increase
# however, it was noticed that removing the words negatively impacted the accuracy of the model. Hence it is placed in comments.

# popouts = [28,13,8,7,6,4,2,1] # added extra based on observation => 4,7,
# popouts = [7]
# for i in popouts:
#   for j in range(len(tf_vectors)):
#     tf_vectors[j].pop(i)

###################################################################################
#################### CLASSIFICATION USING ML ######################################
###################################################################################

randomState = random.randint(0,100)
# randomState = 20
print("random state ->", randomState)
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

# clf = MLPClassifier(max_iter=3000,hidden_layer_sizes=(200,),learning_rate_init=0.01)
clf = RandomForestClassifier(max_depth=6,max_features=None,max_leaf_nodes=9,n_estimators=150)
clf.fit(X_train,y_train)


def hyper_prameter_tune_mlp(X,y):
  mlp_gs = MLPClassifier(max_iter=3000)
  parameter_space = {
      'hidden_layer_sizes': [(200,),(400,),(600,)],
      'learning_rate_init':[0.1,0.01,0.001,0.0001],
      'learning_rate': ['constant'],
  }
  clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
  # clf.fit(X, y) # X is train samples and y is the corresponding labels
  clf.fit(X,y)

  print(clf.best_params_)
  checkAccuracy(clf,y_test,X_test)
# hyper_prameter_tune_mlp(X=X,y=y)

def hyper_parameter_tune_rf(X,y):
  
  param_grid = { 
      'n_estimators': [25, 50, 100, 150], 
      'max_features': ['sqrt', 'log2', None], 
      'max_depth': [3, 6, 9], 
      'max_leaf_nodes': [3, 6, 9], 
  } 
  grid_search = GridSearchCV(RandomForestClassifier(), 
                            param_grid=param_grid) 
  grid_search.fit(X, y) 
  print(grid_search.best_estimator_) 

# hyper_parameter_tune_rf(X=X,y=y)

checkAccuracy(clf,y_test,X_test)
exit()
##################################################################################
################## DIMENSIONALITY REDUCTION ######################################
##################################################################################

y_pred = clf.predict(X_test)
F1_score = accuracy_score(y_test, y_pred)
max_acc = F1_score
print(max_acc)

ogtf = copy.deepcopy(X)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=randomState)
    clf = MLPClassifier(max_iter=3000)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    F1_score = accuracy_score(y_test, y_pred)
    if(F1_score > max_acc):
      popout = i
      acc_increasing = True
      max_acc = F1_score

  if (popout > -1):
    # popout particular column from all the rows
    print(f"popping out {popout}")
    for i in range(len(ogtf)):
      ogtf[i].pop(popout)
      # com_words.pop(com_words.keys()[popout])
  else:
    checkAccuracy(clf,y_test,X_test)

print(max_acc)
print(f"ogtf contains {len(ogtf)} rows and {len(ogtf[0])} columns")

###################################################################################
###################################################################################
###################################################################################

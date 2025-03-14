# program to tune hyperparameters in random forest for the viva dataset.

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

randomState = random.randint(0,100)
# randomState = 20
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

# clf1 = MLPClassifier(max_iter=3000)
# # clf = GradientBoostingClassifier()
clf = RandomForestClassifier()
clf.fit(X_train,y_train)


clf2 = RandomForestClassifier()

from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = clf2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(X_train, y_train)

checkAccuracy(clf,y_test,X_test)
checkAccuracy(rf_random,y_test,X_test)
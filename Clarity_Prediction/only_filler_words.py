# in this experiment, we check which model is best for classifying clarity based on the words used in the answer
# result => RandomForest and MLP classifier provide similar scores. Hence, it is best to make a stacking model using both together.


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
from sklearn.model_selection import train_test_split,cross_val_score
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
 
df = pd.read_csv(dataset,encoding='windows-1252')
wc = Counter()
for answer in df["answer"]:
  answer = str(answer)
  answer = answer.lower()
  wc.update(Counter(word_tokenize(answer)))

# we only consider the most common words out of all
com_words = dict(wc.most_common(34))
# uptil 34th word is considered because the 34th word is "uhh" which is considered to be very important filler word.

vocabulary_count = dict(wc)
vocabulary = list(wc.keys())

# now take each of these words one by one, in order to predict for clarity.
# we use term frequency vectorizer for these 34 words
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
#################### CLASSIFICATION USING ML ######################################
###################################################################################

from imblearn.over_sampling import SMOTE


X = tf_vectors
y = df['clarity'].values.astype('b')

smote=SMOTE(sampling_strategy='minority') 
X,y=smote.fit_resample(X,y)

randomState = random.randint(0,100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=randomState)


clf1 = DecisionTreeClassifier()
clf2 = GaussianNB()
clf3 = MLPClassifier(max_iter=2000)
clf4 = KNeighborsClassifier()
clf5 = AdaBoostClassifier()
clf6 = RandomForestClassifier()
clf7 = svm.SVC()

print("cv : " , cross_val_score(clf1, X, y, cv=5))
print("cv : " , cross_val_score(clf2, X, y, cv=5))
print("cv : " , cross_val_score(clf3, X, y, cv=5))
print("cv : " , cross_val_score(clf4, X, y, cv=5))
print("cv : " , cross_val_score(clf5, X, y, cv=5))
print("cv : " , cross_val_score(clf6, X, y, cv=5))
print("cv : " , cross_val_score(clf7, X, y, cv=5))
print("---------------------------------------")

# clf1.fit(X_train)




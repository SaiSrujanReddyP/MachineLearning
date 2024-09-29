import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
import re
from nltk import word_tokenize
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.datasets import make_classification
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#import math


def checkAccuracy(clf, y_test, X_test):
  y_pred = clf.predict(X_test)
  F1_score = f1_score(y_test, y_pred)
  Precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)

  confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
  confusionMatrixDisplay.plot()
  plt.show()

  # check scores
  print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
  print({"Precision":Precision,"recall":recall,"F1_score":F1_score})

dataset = 'vivadata.csv'

# read data from dataset and place into dataframe
df = pd.read_csv(dataset,encoding='windows-1252')

# vectorize the data using tf-idf
wordset = []
corpus = df["answer"].values.astype('U')
for i in df.index:
  df['answer'].at[i] = re.sub('[^a-z\s]','',corpus[i].lower())
  words = word_tokenize(df['answer'][i])
  for word in words:
    if word not in wordset:
      wordset.append(word)

wordset.sort()

pipe = Pipeline([('count', CountVectorizer(vocabulary=wordset)),('tfid', TfidfTransformer())]).fit(df['answer'])

# tfidfVectors2D contains the vectorized data
tfidfVectors2D = pipe['count'].transform(df['answer']).toarray()

# splitting data into test and train data
X_train, X_test, y_train, y_test = train_test_split(tfidfVectors2D, df['clarity'].values.astype('b'), test_size=0.33, random_state=random.randrange(0,100))

# search = clf.fit(X_train, y_train)
clf = MLPClassifier(random_state=1, max_iter=2000)

# Define the parameter grid
param_dist = {
    'hidden_layer_sizes': [(np.random.randint(1, 100),) for _ in range(20)]  # Generate 20 random layer sizes
}

print("using neural network with best hyperparameters :")
# Using RandomizedSearchCV
search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, scoring='f1', random_state=0)
search.fit(X_train, y_train)
print("best hyperparameters => ", search.best_params_ )
checkAccuracy(search,y_test,X_test)

#print("best parameters : ",search.best_params_)


# usage of svm for this problem
print("using svm")
svmClassifier = svm.SVC().fit(X_train, y_train)
checkAccuracy(svmClassifier,y_test,X_test)

# usage of decision tree
print("using decision tree")
dtree = tree.DecisionTreeClassifier().fit(X_train, y_train)
checkAccuracy(dtree,y_test,X_test)

# naive bayes
print("using naive bayes")
gnb = GaussianNB().fit(X_train, y_train)
checkAccuracy(gnb,y_test,X_test)

# random forest
print("using random forest")
rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
checkAccuracy(rf,y_test,X_test)

# adaboost
print("using adaboost")
adb = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0).fit(X_train, y_train)
checkAccuracy(adb,y_test,X_test)
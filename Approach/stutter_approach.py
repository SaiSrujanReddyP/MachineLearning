import random
import pandas as pd
from collections import Counter
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt

dataset = 'vivadata.csv'

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

df = pd.read_csv(dataset,encoding='windows-1252')
wc = Counter()
for answer in df["answer"]:
  answer = str(answer)
  answer = answer.lower()
  wc.update(Counter(word_tokenize(answer)))

print(wc)

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
  print(term_freq)
  tf_vectors.append(term_freq)


# now, out of these, we use dimensionality reduction to choose the most useful words
# decision tree classifier is used.
clf = DecisionTreeClassifier(random_state=0)

for j in range(len(tf_vectors)):
    tf_vectors[j].pop(13)
for j in range(len(tf_vectors)):
    tf_vectors[j].pop(17)
for j in range(len(tf_vectors)):
    tf_vectors[j].pop(0)
for j in range(len(tf_vectors)):
    tf_vectors[j].pop(16)

X_train, X_test, y_train, y_test = train_test_split(tf_vectors, df['clarity'].values.astype('b'), test_size=0.33, random_state=random.randrange(0,100))

clf.fit(X_train,y_train)




checkAccuracy(clf,y_test, X_test)

y_pred = clf.predict(X_test)
F1_score = f1_score(y_test, y_pred)
max_acc = F1_score
print(max_acc)


ogtf = tf_vectors
for i in range(len(ogtf[0])):
  tf_vectors = ogtf
  for j in range(len(tf_vectors)):
    tf_vectors[j].pop(i)
  X_train, X_test, y_train, y_test = train_test_split(tf_vectors, df['clarity'].values.astype('b'), test_size=0.2, random_state=random.randrange(0,100))

  # now, out of these, we use dimensionality reduction to choose the most useful words
  # decision tree classifier is used.
  clf = DecisionTreeClassifier(random_state=0)
  clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)
  F1_score = f1_score(y_test, y_pred)
  if(F1_score > max_acc):
    print(i)
    print(max_acc)
    max_acc = F1_score
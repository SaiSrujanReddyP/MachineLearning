import numpy as np
#from sklearn.cluster import KMeans
import seaborn as sn
import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import sklearn.metrics
import librosa
import pandas as pd
#from playsound import playsound

file_paths = []
for i in range(1,29):
  file_paths.append("D:\\machineLearningProject\\a" + str(i) + ".wav")


df = pd.read_csv("dataset.csv") # confidence and clarity of each recoring is stored in dataset.

mfccs = []

max_accuracy = 0
preffered_length = 0
max_length = 1200
for f in file_paths:
  signal, sr = librosa.load(f)  #signal contains the amplitude of each sample in the audio with sampling rate of sr

  mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T  

# pad or truncate MFCC so that it can be of uniform size so that it can be placed into the ML model.
  if mfcc.shape[0] < max_length:
      padding = max_length - mfcc.shape[0]
      mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')
  else:
      mfcc = mfcc[:max_length, :]
  mfccs.append(mfcc)

X_flattened = np.array([x.flatten() for x in mfccs])
X_train, X_test, y_train, y_test = train_test_split(X_flattened, df["confidence"].to_list(), test_size=0.2, random_state=random.randint(0,100))

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
def q1():
  F1_score = sklearn.metrics.f1_score(y_test, y_pred)
  Precision = sklearn.metrics.precision_score(y_test, y_pred)
  recall = sklearn.metrics.recall_score(y_test, y_pred)
  confusionMatrixDisplay = sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(y_test, y_pred), display_labels = [0,1])
  confusionMatrixDisplay.plot()
  plt.show()
  print(f'Accuracy: {sklearn.metrics.accuracy_score(y_test, y_pred)}')
  print({"Precision":Precision,"recall":recall,"F1_score":F1_score})


#calculation of MSE
def calcMSE(testData, predictedData):
  error = 0
  for i in range(len(testData)):
    error += (testData[i] - predictedData[i])**2

  return error / len(testData)

#calculation of RMSE 
def calcRMSE(testData, predictedData):
  return math.sqrt(calcMSE(testData,predictedData))
def q2():
  print("MSE score => ", calcMSE(y_test,y_pred))
  print("RMSE score => ", calcRMSE(y_test,y_pred))
  print("MAPE score => ", sklearn.metrics.mean_absolute_error(y_test,y_pred))
  print("R2 score => ", sklearn.metrics.r2_score(y_test,y_pred))

#question 3
def q345():
    
  x = [random.randint(0,10) for i in range(20)]
  y = [random.randint(0,10) for i in range(20)]
  dclass = []
  for i in range(20):
    if(x[i] + y[i] > 10):
      dclass.append("red")
    else:
      dclass.append("blue")

  df = pd.DataFrame()
  df["x"] = x
  df["y"] = y
  df["class"] = dclass
  sn.scatterplot(data=df, x="x", y="y", hue="class")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()

  test_x = [i%100*0.1 for i in range(10000)]
  test_y = [int(i/100)*0.1 for i in range(10000)]

  df2 = pd.DataFrame()
  df2["x"] = test_x
  df2["y"] = test_y
  dclass_pred = []
  knn2 = KNeighborsClassifier(n_neighbors=3)
  knn2.fit(df[["x","y"]],dclass)

  dclass_pred.extend(knn2.predict(df2[["x","y"]]))

  df2["class"] = dclass_pred.copy()

  sn.scatterplot(data=df2, x="x", y="y",hue="class")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()

#question 6 is skipped since we cannot randomly generate test data for our data, it will be entirely random and signify nothing

def q7():
  param_grid = {
      'n_neighbors': list(range(1, 10))  
  }
  randomized_search = RandomizedSearchCV(
      knn,  
      n_iter=100,
      param_distributions=param_grid,   
      cv=5,
      scoring='accuracy', 
      n_jobs=2
  )
  randomized_search.fit(X_train, y_train)
  print(f"Best Parameters: {randomized_search.best_params_}")
  print(f"Best Cross-Validation Score: {randomized_search.best_score_}")


  grid_search = GridSearchCV(
      estimator=knn,
      param_grid=param_grid,
      scoring='accuracy',  # Metric to evaluate the model
      cv=5,  # Number of cross-validation folds
      verbose=1,  # Print progress messages
      n_jobs=2  # Use all available CPUs
  )

  # Fit GridSearchCV
  grid_search.fit(X_train, y_train)

  # Print the best parameters and score
  print("Best parameters found: ", grid_search.best_params_)
  print("Best cross-validation score: ", grid_search.best_score_)

import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score,r2_score,mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
import math

from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans


#vectorization of data
df = pd.read_csv("vivaData.csv")

corpus = df["answer"]
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)
wordset = vectorizer.get_feature_names_out()  
pipe = Pipeline([('count', CountVectorizer(vocabulary=wordset)),('tfid', TfidfTransformer())]).fit(corpus)

tfidfVectors2D = pipe['count'].transform(corpus).toarray()

X_train, X_test, y_train, y_test = train_test_split(tfidfVectors2D, df['clarity'], test_size=0.2, random_state=42)

def TryLinearRegression():
# trying linear regression on data
  X_train, X_test, y_train, y_test = train_test_split(tfidfVectors2D, df['clarity'], test_size=0.2, random_state=42)
  reg = LinearRegression().fit(X_train, y_train)
  y_pred = reg.predict(X_test)
  print("MSE score => ", mean_squared_error(y_test,y_pred))
  print("RMSE score => ", math.sqrt(mean_squared_error(y_test,y_pred)))
  print("MAPE score => ", mean_absolute_error(y_test,y_pred))
  print("R2 score => ", r2_score(y_test,y_pred))
  # extreme values for all these scores show that Linear Regression performs extremely poor for predictions
  i = 0
  for row in df["confidence"]:
    numpy.append(tfidfVectors2D[i],row)
    i += 1
  X_train, X_test, y_train, y_test = train_test_split(tfidfVectors2D, df['clarity'], test_size=0.2, random_state=42)
  # trying linear regression again with one more attribute, confidence
  reg = LinearRegression().fit(X_train, y_train)
  y_pred = reg.predict(X_test)
  print("MSE score => ", mean_squared_error(y_test,y_pred))
  print("RMSE score => ", math.sqrt(mean_squared_error(y_test,y_pred)))
  print("MAPE score => ", mean_absolute_error(y_test,y_pred))
  print("R2 score => ", r2_score(y_test,y_pred))
  # similar extreme values appears again, hence proving that linear regression models won't work.

# TryLinearRegression()

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(tfidfVectors2D)

print("silhouette score = ",silhouette_score(tfidfVectors2D, kmeans.labels_))
print("CH index = ",calinski_harabasz_score(tfidfVectors2D, kmeans.labels_)) 
print("DB index = ",davies_bouldin_score(tfidfVectors2D, kmeans.labels_))
# high value for CH index and low value for DB index indicate that there is somewhat good clustering

sil_scores = []
CH_scores = []
DB_scores = []

distorsions = []
for k in range(2, 20):  
  kmeans = KMeans(n_clusters=k).fit(tfidfVectors2D) 
  distorsions.append(kmeans.inertia_) 
  sil_scores.append(silhouette_score(tfidfVectors2D, kmeans.labels_))
  CH_scores.append(calinski_harabasz_score(tfidfVectors2D, kmeans.labels_))
  DB_scores.append(davies_bouldin_score(tfidfVectors2D, kmeans.labels_))
plt.plot([i for i in range(2,20)],distorsions)
plt.show()

plt.plot([i for i in range(2,20)],sil_scores)
plt.ylabel("silhouette scores")
plt.show()
plt.plot([i for i in range(2,20)],CH_scores)
plt.ylabel("CH index")
plt.show()
plt.plot([i for i in range(2,20)],DB_scores)
plt.ylabel("DB index")
plt.show()

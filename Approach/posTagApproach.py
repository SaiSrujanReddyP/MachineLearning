import nltk
nltk.download('treebank')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import treebank
import matplotlib.pyplot as plt
import re
import pandas as pd


dataset = 'vivadata.csv'

# read data from dataset and place into dataframe
df = pd.read_csv(dataset,encoding='windows-1252')

transitions = {}

f = open("text.txt",encoding='utf-8')
words = f.read()
words = re.sub('[\,\"\'\-]','',words)
sentences = re.split('\. ',words)
f.close()

print(words)
for sentence in sentences:
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)

    for i in range(len(tagged_tokens) - 1):
      try:
        if tagged_tokens[i + 1][1] in transitions[tagged_tokens[i][1]]:
          #print("detected")
          continue
        else:
          transitions[tagged_tokens[i][1]].append(tagged_tokens[i + 1][1])
      except KeyError:
        transitions[tagged_tokens[i][1]] = [tagged_tokens[i + 1][1]]

FP = []
FPt = []
TP = []
TPt = []
TN = 0
FN = 0
for j in df.index:

  sentence = df['answer'][j]
  sentence = str(sentence).lower()
  sentence = re.sub('[\,\"\'\-]','',sentence)
  tokens = nltk.word_tokenize(sentence)
  tagged_tokens = nltk.pos_tag(tokens)
  unclear = 0
  for i in range(len(tagged_tokens) - 1):
    try:
      if tagged_tokens[i + 1][1] not in transitions[tagged_tokens[i][1]]:
        unclear += 1
    except KeyError:
      unclear += 1

  if df['clarity'][j] == 0: # is unclear
    if unclear == 0:        # predicted as clear
      FN += 1
    else:                   # predicted as unclear
      print("true detected -> ",unclear/len(tokens))
      TP.append(unclear)
      TPt.append(len(tokens))
  else:                     # is clear
    if unclear > 0:         # predicted as unclear
      print("false positive detected -> ",unclear/len(tokens))
      FP.append(unclear)
      FPt.append(len(tokens))
    else:                   # predicted as clear  
      TN += 1

print("TP => ", len(TP))
print("FP => ", len(FP))
print("TN => ", TN)
print("FN => ", FN)


plt.scatter(x=FP,y=FPt)
plt.show()
plt.scatter(x=TP,y=TPt)
plt.show()


exit()



# test
sentence = input("enter a sentence : ")
tokens = nltk.word_tokenize(sentence)
tagged_tokens = nltk.pos_tag(tokens)
unclear = 0
for i in range(len(tagged_tokens) - 1):
  try:
    if tagged_tokens[i + 1][1] not in transitions[tagged_tokens[i][1]]:
      unclear += 1
  except KeyError:
    unclear += 1

print("unclarity measure = ", unclear)
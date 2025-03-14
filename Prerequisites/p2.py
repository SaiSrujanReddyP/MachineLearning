import random
import re
from datetime import datetime
import pandas as pd


matrix = [[0],[0],[0],[0]]
for i in matrix:
	for j in range(3):
		i += [0]

for i in range(len(matrix)):
	for j in range(len(matrix[i])):
		matrix[i][j] = random.randrange(0,100)

print(str(matrix))

singleMatrix = matrix[0]

for i in matrix:
	singleMatrix += i

print("converted single matrix -> " + str(singleMatrix))

string1 = "I am a great learner. I am going to have an awesome life."

print(f"am exists in the string {string1.count('am')} times")

s2 = "I work hard and shall be rewarded well"

s3 = string1 + s2

s3array = re.split(r'[\s|.]',s3)
print(str(s3array))

s3array.remove("am")
s3array.remove("I")
s3array.remove("to")
s3array.remove("and")

for i in s3array:
	if(len(i) > 6):
		s3array.remove(i)

print(str(s3array))

dat = "01-JUN-2021"
dat = datetime.strptime(dat,"%d-%b-%Y")
print(dat)

data = "CITY STATE PINCODE BENGALURU KA 560001 CHENNAI TN 600001 MUMBAI MH 400001 MYSURU KA 570001 PATNA BH 800001 JAMMU JK 180001 GANDHI_NAGAR GJ 382001 HYDERABAD TS 500001 ERNAKULAM KL 682001 AMARAVATI AP 522001"
dataArray = data.split(' ')
file = open(r"data.csv","w")

c = 0
for i in dataArray:
	file.write(i)
	c += 1
	if(c % 3 == 0 and c != 0):
		file.write('\n')
	else:
		file.write(',')
file.close()

df = pd.read_csv("data.csv")
print(df)


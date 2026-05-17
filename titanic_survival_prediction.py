import pandas as pd #Required Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = sns.load_dataset("titanic") #Loading Dataset

df = df[["survived","pclass","sex","age","fare"]] #Only using the needed Columns
print(df.head(10))
print(df.shape)
print(df.columns)

df["age"] = df["age"].fillna(df["age"].mean()) #Filling the empty slots

df["sex"] = df["sex"].map({"male":0,"female":1}) #Converting male, female into numbers

print(df.head(10))

X = df[["pclass","sex","age","fare"]] #Selecting Feature
y = df["survived"] #Selecting Target

X_train, X_test, y_train, y_test = train_test_split( #Dividing the data into Training data and Testing data
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression() #Creating a Model
model.fit(X_train,y_train) #Training the Model

predictions = model.predict(X_test) #Predicting
print (predictions[:10])

print("Accuracy :", accuracy_score(y_test,predictions))#Calculating the Accuracy score

print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

df.groupby("sex")["survived"].mean().plot(kind="bar") #BarGraph of Survival Rate Of Male & Female
plt.xlabel("Sex (0=male, 1=female)")
plt.ylabel("Survival Rate")
plt.title("Survival Rate Of Male & Female")
plt.show()
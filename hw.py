import numpy as np
import matplotlib.pyplot as mp 
import pandas as pd
import seaborn as sb

data= pd.read_csv("adult.csv")
print(data.info())
data.columns=['age','workclass','fnlwgt','education','educational-num','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss',"hours-per-week","native-country","income"]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['workclass']=le.fit_transform(data['workclass'])
data['education']=le.fit_transform(data['education'])
data['marital-status']=le.fit_transform(data['marital-status'])
data['occupation']=le.fit_transform(data['occupation'])
data['relationship']=le.fit_transform(data['relationship'])
data['race']=le.fit_transform(data['race'])
data['gender']=le.fit_transform(data['gender'])
data['native-country']=le.fit_transform(data['native-country'])
data['income']=le.fit_transform(data['income'])

X=data[['age','workclass','fnlwgt','education','educational-num','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss',"hours-per-week","native-country"]]
y=data['income']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

from sklearn.ensemble import RandomForestClassifier
#n-estiomatorr=1-num of trees
c=RandomForestClassifier(n_estimators=100)
c.fit(X_train,y_train)
y_pred=c.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy',random_state=0)
dt.fit(X_train,y_train)
y_pred1=dt.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
sb.heatmap(cm,annot=True,fmt="d")
mp.title("CONFUSION MATRIX 1 random forest ")
mp.xlabel("actual")
mp.ylabel("predicted")
mp.show()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred1)
sb.heatmap(cm,annot=True,fmt="d")
mp.title("CONFUSION MATRIX 2 decision tree ")
mp.xlabel("actual")
mp.ylabel("predicted")
mp.show()

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_pred,y_test)
print("random forest",acc)
acc1=accuracy_score(y_pred1,y_test)
print("decision tree",acc1)

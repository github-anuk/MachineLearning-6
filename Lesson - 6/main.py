#RANDOMFOREST - classification 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb

data=pd.read_csv("student-mat.csv")
print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["school"]=le.fit_transform(data["school"])
data["sex"]=le.fit_transform(data["sex"])
data['famsize']=le.fit_transform(data["famsize"])
data['Pstatus']=le.fit_transform(data["Pstatus"])
data["Mjob"]=le.fit_transform(data["Mjob"])
data["Fjob"]=le.fit_transform(data["Fjob"])
data['reason']=le.fit_transform(data["reason"])
data['guardian']=le.fit_transform(data["guardian"])
data["schoolsup"]=le.fit_transform(data["schoolsup"])
data["famsup"]=le.fit_transform(data["famsup"])
data['paid']=le.fit_transform(data["paid"])
data['activities']=le.fit_transform(data["activities"])
data["nursery"]=le.fit_transform(data["nursery"])
data["higher"]=le.fit_transform(data["higher"])
data['internet']=le.fit_transform(data["internet"])
data['romantic']=le.fit_transform(data["romantic"])

data.drop('G1',axis=1,inplace=True)
data.drop('G2',axis=1,inplace=True)
print(data.head())

X=data[["school","sex",'age','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences']]
y=data["G3"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

from sklearn.ensemble import RandomForestClassifier
#n-estiomatorr=1-num of trees
c=RandomForestClassifier(n_estimators=100)
c.fit(X_train,y_train)
y_pred=c.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
sb.heatmap(cm,annot=True,fmt="d")
mp.title("CONFUSION MATRIX")
mp.xlabel("actual")
mp.ylabel("predicted")
mp.show()
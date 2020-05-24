
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
import pickle
import numpy as np

filename1 =sys.argv[1]
filename2=sys.argv[2]

def dataprep(filename1,filename2):
    df1 = pd.read_table(filename1,delimiter=',', header=None)
    df2= pd.read_table(filename2,delimiter=',', header=None)
    new_header1 = df1.iloc[0]
    df1 = df1[1:]
    new_header2 = df2.iloc[0]
    df2 = df2[1:]
    df1.columns = new_header1
    df2.columns = new_header2
    X1=df1
    X2=df2
    X1 = df1.loc[:, df1.columns != 'class']
    y = df1['class'].to_frame()
    X_enc1 = pd.get_dummies(X1)
    X_enc2 = pd.get_dummies(X2)
    X_enc1, X_enc2 = X_enc1.align(X_enc2, join='left', axis=1)
    X_enc2 = X_enc2.apply (pd.to_numeric, errors='coerce')
    X_enc2 = X_enc2.fillna(0)
    print('\ntraining is\n',X_enc1.head())
    print('\ntraining is\n',X_enc2.head())
    scaler = StandardScaler()
    X_std1 = scaler.fit_transform(X_enc1)
    X_std2 = scaler.fit_transform(X_enc2)
    le = LabelEncoder()
    y_enc = le.fit_transform(y.values.ravel())
    print('y_enc is',y_enc)
    return X_std1,y_enc,X_std2

def train(X_std1,y_enc1):
    classifiers=[]
    X_train, X_test, y_train, y_test = train_test_split(X_std1,y_enc1,test_size=0.3, shuffle=True)
    model1 = xgboost.XGBClassifier()
    classifiers.append(model1)
    model2 = SVC()
    classifiers.append(model2)
    model3 = DecisionTreeClassifier()
    classifiers.append(model3)
    model4 = RandomForestClassifier()
    classifiers.append(model4)
    i=0
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred= clf.predict(X_test)
        if(i==0):
            savefile_mod1 = 'xgboost_model.sav'
            pickle.dump(clf, open(savefile_mod1, 'wb'))
        elif(i==1):
            savefile_mod2 = 'svm_model.sav'
            pickle.dump(clf, open(savefile_mod2, 'wb'))
        elif(i==2):
            savefile_mod3 = 'Decision_tree_model.sav'
            pickle.dump(clf, open(savefile_mod3, 'wb'))
        else:
            savefile_mod4 = 'Random_forest_model.sav'
            pickle.dump(clf, open(savefile_mod4, 'wb'))
        i+=1
        acc = accuracy_score(y_test, y_pred)
        print("\nAccuracy of %s method is %s"%(clf, acc))
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix of %s method  is %s"%(clf, cm))
        cl=classification_report(y_test, y_pred)
        print('classification_report is\n',cl)

def test(test_X):
    loaded_model1 = pickle.load(open('xgboost_model.sav', 'rb'))
    pred1= loaded_model1.predict(test_X)
    loaded_model2 = pickle.load(open('svm_model.sav', 'rb'))
    pred2= loaded_model2.predict(test_X)
    loaded_model3 = pickle.load(open('Decision_tree_model.sav', 'rb'))
    pred3= loaded_model3.predict(test_X)
    loaded_model4 = pickle.load(open('Random_forest_model.sav', 'rb'))
    pred4= loaded_model4.predict(test_X)
    return np.array(pred1),np.array(pred2),np.array(pred3),np.array(pred4)

X_tr,y_tr,X_tes=dataprep(filename1,filename2)
pred=np.zeros(4,dtype=int)
train(X_tr,y_tr)
pred1,pred2,pred3,pred4=test(X_tes)
result=[]
for i in range(0 ,len(pred1)):
    pred[0]=pred1[i]
    pred[1]=pred2[i]
    pred[2]=pred3[i]
    pred[3]=pred4[i]
    print(pred)
    sum_pred=np.sum(pred)
    if(sum_pred>2):
        mode_value=1
    elif(sum_pred<2):
        mode_value=0
    else :
        mode_value=1
    print('\n mode of obs is:',mode_value)
    if(mode_value==0):
        result.append('E')
    elif(mode_value==1):
        result.append('P')
print('result is\n',result)
with open('Question1.txt', 'w') as f1:
    for j in range (0,len(pred1)):
        line=result[j]+"\n"
        f1.write(line)

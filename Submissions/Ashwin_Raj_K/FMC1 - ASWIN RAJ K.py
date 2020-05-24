import numpy as np
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

my_data = np.genfromtxt('train.csv', delimiter=',',dtype=str)
my_data_converted = np.array([[ord(element) for element in row] for row in my_data[1:,:]])


X = my_data_converted[:,1:]
y = my_data_converted[:,0]


polynomial_svm_clf = Pipeline((("poly_features", PolynomialFeatures(degree=3)),
                               ("scaler", StandardScaler()),
                               ("svm_clf", LinearSVC(C=10, loss="hinge"))))


polynomial_svm_clf.fit(X,y)

my_data_test = np.genfromtxt('Q1_Mushroom_test.csv', delimiter=',',dtype=str)
my_data_converted_test = np.array([[ord(element) for element in row] for row in my_data_test[1:,:]])

y_pred = np.array([])
for i in polynomial_svm_clf.predict(my_data_converted_test):
    if(i==112):
        y_pred = np.append(y_pred,'p')
    else:
        y_pred = np.append(y_pred, 'e')

np.savetxt("Q1Answer.txt",y_pred.reshape(-1,1),fmt="%s")
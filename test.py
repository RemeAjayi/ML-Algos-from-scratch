from KNN import KNN
import pandas as pd
import numpy as np 
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def main():
    data = datasets.load_iris()
    mm_scaler = preprocessing.MinMaxScaler()

    X = mm_scaler.fit_transform(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    model = KNN()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test, 12, True) #k ~ N ^ 1/2
    
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)


if __name__ == "__main__":
    main()

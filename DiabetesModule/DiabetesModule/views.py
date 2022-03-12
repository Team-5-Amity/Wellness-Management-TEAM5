import numpy as np
from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):

    df = pd.read_csv(r'C:\Users\SUBHADEEP\PycharmProjects\pythonProject\Diabetes Module\diabetes.csv')
    x = df.iloc[:, 0:8]
    y = df.iloc[:, 8]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""
    if pred == [1]:
        result1 = " Positive"
    else:
        result1 = " Negative"

    return render(request, "predict.html", {"result2": result1})

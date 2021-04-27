from django.shortcuts import render
import csv,io
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt, mpld3
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create your views here.
def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def predict(request):
    # if request.POST.get('action') == 'post':
    if request.method == 'POST':
        name= request.POST.get('name')
        rbp= request.POST.get('rbp')
        print("name ",name)
        print("rbp ",rbp)
        age = request.POST.get('age')
        print("Age")
        print(age)
        # state_name = request.POST['search_text'].capitalize()
        heartdata = pd.read_csv("heart.csv")
        heartdata.head()
        print(heartdata.head())
        heartdata_X = heartdata
        heartdata_X_train = heartdata_X[:]
        heartdata_X_test = heartdata_X[-1000:]

        heartdata_Y_train= heartdata.target[:]
        heartdata_Y_test = heartdata.target[-1000:]

        model = linear_model.LinearRegression()
        model.fit(heartdata_X_train,heartdata_Y_train)
        heartdata_Y_predicted = model.predict(heartdata_X_test)
        
        fig, ax = plt.subplots()
        plt.plot(heartdata_X_test)
        plt.plot(heartdata_Y_predicted)
        plt.legend(['Test Data', 'Linear Regression Predictions'])
        fig.savefig("static/images/1.png",dpi = 70)

        print("Means S Error: ",mean_squared_error(heartdata_Y_test,heartdata_Y_predicted))
        print("Weights: ", model.coef_)
        print("Intersepts: ", model.intercept_)

        # svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
        # svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
        # svm_pred = svm_confirmed.predict(future_forcast)

        # # check against testing data
        # svm_test_pred = svm_confirmed.predict(X_test_confirmed)
        return render(request, 'predict.html')
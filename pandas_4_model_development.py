#########################
""" MODEL DEVELOPMENT """
#########################

"""
Model: can be thought of:
    - as a mathematical equation used to predict a value given one or more values
    - relating one or more independant variables to dependant variables

Usually, the more data and more relevant data you provide the model, the more accurate the result is.
"""

######################################################
""" LINEAR REGRESSION & MULTIPLE LINEAR REGRESSION """
######################################################

"""
(Simple) Linear Regression (SLR) will refer to one independent variable to make a prediction:
    -> y = b0 + b1x
    -> will store data points of x in NumPy arrays and this gives us y output
    -> training points --> fit or train model to get params. --> use params. in model --> we have model

Multiple Linear Regression (MLR) will refer to multiple independent variables to make a prediction.
"""

###############################################
""" FITTING A SIMPLE LINEAR MODEL ESTIMATOR """
###############################################

# import linear_model from scikit-learn
from sklearn.linear_model import LinearRegression

# create a Linear Regression Object using constructor
lm = LinearRegression()

# define predictor and target variables
X = df[["highway-mpg"]]  # predictor
Y = df[["price"]]  # target

# use lm.fit(X,Y) to fit model (constant and gradient, b0, b1)
lm.fit(X,Y)

# obtain prediction
Yhat = lm.predict(X)  # the output is an array

# view intercept (b0)
    # lm.intercept_

# view slop (b1)
    # lm.coef_

#################################################
""" FITTING A MULTIPLE LINEAR MODEL ESTIMATOR """
#################################################

"""
MLR used to explain the relationship between:
    - one continuous target variable (Y)
    - two or more predictor variables (X)
    
For example:
    - b0: intercept (X=0)
    - b1: coeff. of param. x1
    - b2: coeff. of param. x2
    - etc.
    
If there's two predictors, x1 and x2, they can be plotted on a 2d-plane and Y will be mapped on the
vertical direction.
"""

# extrtact 4 predictor variables and score in Z
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]

# train the model
lm.fit(Z, df["price"])

# obtain prediction
Yhat2 = lm.predict(X)  # outputs array with 4 columns

############################################
""" MODEL EVALUATION USING VISUALISATION """
############################################

# regression plot using seaborn

import seaborn as sns

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

# example
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

# correlation method
df[["peak-rpm","highway-mpg","price"]].corr()

"""
Residual plots, plot the difference between target and actual values, which allow us to obtain
insight into which model is best.

For example, a residual plot with 0 mean, distributed equally with similar variance means we can
use the Linear Model.

If there is curvature we must use a non-linear model.
"""

# residual plot using seaborn

sns.residplot(df["highway-mpg"], df["price"])

# example
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()

"""
Distribution plots count the predicted value against the actual value.
"""

# distribution plot using seaborn
# hist=False if we want distribution (continuous data)

axl = sns.distplot(df["price"], hist=False, color = "r", label="Actual Value")

sns.distplot(Yhat, hist=Flase, color="b", label="Fitted Values", ax=axl)

#########################################
""" POLYNOMIAL REGRESSION & PIPELINES """
#########################################

"""
Polynomial regression is used to describe curvilinear relationships.

Curvilinear relationship: squaring or setting higher-order terms of the predictor variables.
    - Quadratic (2nd order)
    - Cubic (3rd order)
"""

# polynomial regression in NumPy
# 3rd order
f = np.polyfit(x,y,3)
p = np.polyld(f)
print(p)


# for polynomial regression with more than one dimension we use scikit
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2, include_bias=False)  # constructor
x_polly = pr.fit_transform(x[["horsepower", "curb-weight!"]])  # transform


# normalise features for higher orders
from sklearn.preprocessing import StandardScaler
SCALE = StandardScaler()
SCALE.fit(x_data[["horsepower", "highway-mpg"]])
x_scale = SCALE.transform(x_data[["horsepower", "highway-mpg"]])


"""
We can use a pipeline library to sequentially performs the steps we need to get a prediction.

Steps to getting a prediction:
    Normalisation -> Polynomial transform -> Linear regression
"""

# first import all modules
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# create a list of tuples (arg1 = estimator, arg2 = model constructor
Input = [("scale", StandardScaler()), ("polynomial", PolynomialFeatures(degree=2)), ("mode", LinearRegression())]
pipe = Pipeline(Input)  # pipeline constructor

# train the pipeline
Pipe.fit(df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]],y)
yhat = Pipe.predict(X[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])

#########################################
""" MEASURES FOR IN-SAMPLE EVALUATION """
#########################################

"""
A way to numerically determine how good the model fits on dataset.

Two measures to determine fit of a model:
    - Mean Squared Error (MSE)
        - mean of the difference of points squared
        - MSE of MLR is smaller than MSE of SLR
        - Polynomial regression will have smaller MSE than Linear regression
    - R-squared (R^2) (Coefficient of determination)
        - measures how close data is to fitted regression line
        - R^2 = (1-(MSE of regression line)/(MSE of average of data))
        - R^2 generally b/w 0 and 1 (can be negative)
        - R^2 close to 0.1 is good
"""

# MSE from scikit
from sklearn.metrics import mean_squared_error
mean_squared_error(df["price"], Y_predict_simple_fit)  # actual target value, predicted target value


# R-squared
X = df[["highway-mpg"]]
Y = df[["price"]]
lm.fit(X,Y)
lm.score(X,Y)

###################################
""" PREDICTION & DECISION MAKING"""
###################################

# use NumPy function arrange to generate sequence from 1 to 100
import numpy as np
new_input = np.arrange(1, 101, 1).reshape(-1, 1)

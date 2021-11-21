#####################################
""" MODEL EVALUATION & REFINEMENT """
#####################################

"""
We split data into parts used to train (fit) the model and to test the model.
"""

# splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

# x_data: features or independent variables
# y_data: dataset target (df["price"]
# test_size: percentage of data for testing (30% in above example)
# random_state: number generator used for random sampling

"""
There is 'generalisation error' when we choose a certain percentage for training/testing.

To overcome issues with accuracy/precision of data, we can use 'cross-validation'.

Cross-validation: a resampling procedure used to evaluate machine learning models on a limited 
data sample. The procedure has a single parameter called k that refers to the number of groups that 
a given data sample is to be split into.
"""

# apply cross-validation function
from sklearn.model_selection import cross_val_score
score = cross_val_score(lr, x_data, y_data, cv=3)

# lr: type of model (linear regression)
# x_data: predictor
# y_data: target
# cv: number of partitions

# apply cross-validation prediction function
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lr2e,x_data, y_data, cv=3)


#####################################################
""" OVER-FITTING, UNDER-FITTING & MODEL SELECTION """
#####################################################

"""
Under-fitting: model is too simple to fit data

Over-fitting: model is too flexible for the data

Training Error decreases with and increase to the order of the polynomial. Test Error decreases 
until an inflection point and then increases with an increase to the order of the polynomial. Anything
on the left of the inflection is under-fitted and anything to the right is over-fitted, therefore,
the best model is the one with order equal to the inflection point.
"""

# we can find R-squared for many orders of the polynomial
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[["horsepower"]])
    x_test_pr = pr.fit_transform(x_test[["horsepower"]])
    lr.fit(x_train_pr, y_train)
    Rsqu_test.append(lr.score(x_test_pr, y_test))


########################
""" RIDGE REGRESSION """
########################

"""
'Ridge regression' is a regression that is employed in a Multiple regression model when 
'Multicollinearity' occurs. Multicollinearity is when there is a strong relationship among the 
independent variables. Ridge regression is very common with polynomial regression. 

Ridge regression controls the magnitude of the coefficients in large order polynomials using
a parameter called 'Alpha'.

Large Alpha significantly decreases the magnitude of the coefficients but also makes it underfit.
"""

# ridge regression
from sklearn.linear_model import Ridge
RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(X,y)
Yhat = RidgeModel.predict(X)


###################
""" GRID SEARCH """
###################

"""
- Hyperparameters are terms such as Alpha in Ridge Regression.

- Scikit-learn has a means of automatically iterating over these hyperparameters using 
cross-validation which is called 'Grid Search'.

- Grid search calculates the R^2 for different hyperparameters.
"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

parameters1 = [{"alpha":[0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, parameters1, cv=4)
Grid1.fit(x_data[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_data)
Grid1.best_estimator_
scores = Grid1.cv_results_
scores["mean_test_score"]

###########################
""" DATA PRE-PROCESSING """
###########################
import numpy as np
import pandas as pd

"""
'Data Pre-Processing' is the process of converting or mapping data from its initial, raw, form
to another format in order to prepare the data for further analysis. Also known as 'Data-Cleaning'
or 'Data-Wrangling'. For example:
                                 - identifying and handling missing values
                                 - formatting
                                 - normalisation
                                 - binning
                                 - converting categorical variables into numerical
"""

##############################
""" DATAFRAME MANIPULATION """
##############################

# df["columnname"]  # to access a specific column
# df["columnname"] = df["columnname"} + 1  # adds 1 to current value

###################################
""" DEALING WITH MISSING VALUES """
###################################

"""
Below are some ways to deal with missing values:

- Check with the data collection source
- Drop the missing values
    - drop the variable
    - drop the specific entry
- Replace the missing values
    - replace with average (of similar data points)
    - replace by frequency (if categorical data)
    - replace it based on other functions
- Leave it as missing
"""

# we can identify missing data
missing_data = df.isnull()  # gives boolean value if there is missing data
missing_data.head(5)

# using a for loop we can find the number of missing values
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

###########################
""" DROP MISSING VALUES """
###########################

# how to drop missing values in Pandas
# 'axis=0' to drop entire row, 'axis=1' to drop entire column
# using 'inplace = True' makes the change directly
dataframe.dropna()

# this will not modify the dataframe
df = df.dropna(subset=["price"], axis=0)

# this will modify the dataframe directly
df.dropna(subset=["price"], axis=0, inplace=True)

##############################
""" REPLACE MISSING VALUES """
##############################

# how to replace missing values in Pandas
dataframe.replace(missing_value, new_value)

# for example we can replace NaN value with the average for that subset
# if 'inplace' is not set, there are no changes to the dataframe
mean = df["normalized-losses"].mean()  # calculate average
df["normalized-losses"].replace(np.nan, mean, inplace=True)  # replace NaN values with avg

#######################
""" COUNTING VALUES """
#######################

# to see which values are present in a particular column
df['num-of-doors'].value_counts()

# to see which value is the most common automatically
df['num-of-doors'].value_counts().idxmax()

#######################
""" DATA FORMATTING """
#######################

"""
As data can be collected from various sources, we need to bring the data up to a common
standard of expression to allow meaningful comparisons.
"""

# converting a column in 'mpg' to 'L/100km'
df["city-mpg"] = 235/df["city-mpg"]
# renaming column by modifying dataset
df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True)

# identifying and converting data types
df.dtypes(["price"])  # original type is 'object'
df["price"] = df["price"].astype("int")  # converting type

##########################
""" DATA NORMALISATION """
##########################

"""
Data may come in a de-normalised form where the data vary in range vastly for example.

If data is de-normalised, it can skew statistical analyses and give results that are not truly
representative.

Consider variables 'age' and 'income'. The numerical value of income is orders of magnitude
higher than age, which means if a liner-regression is performed, it will be far too heavily favoured
with income, even though it may not be the most important factor.

Below are some examples on methods to normalise data:

- 'Simple Feature Scaling' (value ranges from 0 -> 1):
    xnew = xold/xmax
    
- 'Min-Max' (value ranges from 0 -> 1):
    xnew = (xold - xmin)/(xmax - xmin)
    
- 'Z-score' (value ranges from -3 -> 3):
    xnew = (xold - mu)/sigma
"""

# example of using 'Simple Feature Scaling' to normalise a column
df["length"] = df["length"]/df["length"].max()

# example of using 'Min-Max' to normalise a column
df["length"] = (df["length"] - df["length"].min())/(df["length"].max() - df["length"].min())

# example of using 'Z-score' to normalise a column
df["length"] = (df["length"] - df["length"].mean())/df["length"].std()

###############
""" BINNING """
###############

"""
'Binning' is simply putting values in groups. You can convert numeric into categorical values.
You can group a set of numerical values into a set of bins.
"""

#  creating bins of equal width for price
bins = np.linspace((min(df["price"]), max(df(["price"]), 4)  # using NumPy function 'linspace'
group_names = ["Low", "Medium", "High"]  # setting the bin names
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names, include_lowest=True)  # Pandas 'cut'

##########################
""" PLOTTING HISTOGRAM """
##########################

import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

########################################
""" PLOTTING HISTOGRAM AFTER BINNING """
########################################

import matplotlib as plt
from matplotlib import pyplot
binss = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], binss, labels=group_names, include_lowest=True )
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

#################################################################
""" TURNING CATEGORICAL VARIABLES INTO QUANTITATIVE VARIABLES """
#################################################################

"""
Statistical models cannot take categorical variables as input, so we convert them to a numeric form
so we can perform analysis.

For example, if we have a column called 'Fuel' which take object values of 'gas' and 'diesel',
we can use a technique called 'One-Hot Encoding' to set 'dummy-variables' where one categorical 
variable is set to 1, and the other to 0. If car A using gas, gas is set to 1 and diesel is set to 0.
"""

# setting dummy-variable
pd.get_dummies(df["fuel"])  # automatically created two new columns with dummy-variables

##########################
""" MERGING DATAFRAMES """
##########################

# merge the new dataframe to the original dataframe
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)

# save the new csv
df.to_csv('clean_df.csv')

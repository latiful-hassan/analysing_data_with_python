#################################
""" EXPLORATORY DATA ANALYSIS """
#################################

"""
Exploratory Data Analysis (EDA) is the preliminary step in data analysis to:
    - summarise main characteristics of data
    - gain understanding of data
    - uncover relationships between variables
    - extract important variables
"""

##############################
""" DESCRIPTIVE STATISTICS """
##############################

"""
We can describe basic features of the data set by giving short summaries using the 'describe()'
method for numeric values. 

We can use the 'value_counts()' method to understand how many units of each characteristic or 
variable we have.

The default setting of describe() does not include type object. We can change this with an
'include' argument.
"""

# the describe method
df.describe()  # shows count, mean, std, min, quartiles (25%, 50%, 75%), max

# the value_counts() method
drive_wheel_counts = df["drive-wheels"].value_counts().to_frame()  # convert series to dataframe
drive_wheel_counts.rename(columns = {"drive-wheels": "value_counts"}, inplace=True)  # rename

# use describe to include object data type
df.describe(include=['object'])

###########################
""" PLOTS USING SEABORN """
###########################

# box plot code (shows quartile data)
sns.boxplot(x="drivewheels", y="price", data=df)

# scatter plot code (shows relationship between two variables)
y = df["price"]
x = df["drive-wheels"]
plt.scatter(x,y)
plt.title("Scatter Plot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")

#########################
""" GROUPBY IN PYTHON """
#########################

"""
We can group data by using the 'groupby()' method:
    - can be applied to categorical variables
    - group data in categories
    - group by single or multiple variables
"""

# first check how many unique categories there are
df['drive-wheels'].unique()

# groupby() in Pandas
dataframe.groupby()

# example of using Groupby()
df_test = df[["drive-wheels", "body-style", "price"]]  # select columns
df_grp = df_test.groupby(["drive-wheels", "body-style"], as_index=False).mean()  # group by mean

"""
Pivot tables have one variable displayed along the columns and another along the rows.
"""

# create a pivot table from the above data to make visualisation easier
df_pivot = df_grp.pivot(index="drivewheels", columns="body-style")

"""
Heatmap plot takes a rectangular grid of data and assigns color of intensity based on its value.
"""

# creating heatmap
plt.pcolor(df_pivot, cmap="RdBu")  # red/blue color scheme
plt.colorbar()
plt.show()

###################
""" CORRELATION """
###################

"""
Correlation measure to what extent different variables are interdependent. Correlation does not
imply causation.
"""

# to calculate correlation between variables of type 'int64' op 'float64'
df.corr()
df[['bore','stroke','compression-ratio','horsepower']].corr()  # example

# creating a regression line (seaborn)
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

###########################
""" PEARSON CORRELATION """
###########################

"""
Pearson correlation allows us to measure the strength of correlation between two features:
    - correlation coefficient
    - P-value

Correlation Coefficient:
    - close to +1: large positive relationship
    - close to -1: large negative relationship
    - close to 0: no relationship
    
P-value:
    - P-value < 0.001   strong certainty
    - P-value < 0.05    moderate certainty
    - P-value < 0.1     weak certainty
    - P-value > 0.1     no certainty
    
Strong correlations in either positive or negative come from:
    - correlation coefficient close to 1/-1
    - P-value less < 0.001
"""

# calculate Person correlation using SciPy

from scipy import stats

pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])

#############
""" ANOVA """
#############

"""
'ANOVA', or, Analysis of Variance is a statistical method used to test whether there are 
significant differences between the means of two or more groups.

ANOVA returns two parameters:
    - F-test score: ANOVA assumes the means of all groups are the same, calculates how much the
      actual means deviate from the assumption, and reports it as the F-test score. A larger score 
      means there is a larger difference between the means
      
    - P-value: P-value tells how statistically significant our calculated score value is
    
If our variable is strongly correlated with the variable we are analyzing, we expect ANOVA to return 
a sizeable F-test score and a small p-value.
"""

# example of ANOVA with respect to if 'drive-wheels' affect 'price'

# first group the data
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

# obtain values of the method group using 'get_group'
# grouped_test2.get_group('4wd')['price']

# use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

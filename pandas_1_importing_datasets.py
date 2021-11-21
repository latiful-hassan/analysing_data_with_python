###################
"""IMPORTING CSV"""
###################

import pandas as pd  # importing pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"  # enter url

# df = pd.read_csv(url)  # creating dataframe
df = pd.read_csv(url, header=None)  # 'read_csv' assumes headers, we can set for no headers as below

# df.head(n)  # this gives you top n rows of the data frame
# df.tail(n)  # this gives you bottom n rows of the data frame

print(df.head(5))  # to save time use head/tail

"""
The data printed from above automatically uses integers as headers as we set 'header=none'.
We can manually set headers by created a variable with the column headers and then setting 'df.columns'
"""

headers = ["..."]  # enter headers
df.columns = headers  # setting headers

"""
We can save/export our modified datasets by defining a 'path' variable and then using 'df.to_csv()'
"""

path = "C:/Windows ... "
df.to_csv(path)

"""
Below is shown the different formats we can use and what syntax allows reading/saving:

###################################################
Data Format         Read                Save
###################################################
csv                 pd.read_csv()       df.to_csv()
json                pd.read_json()      df.to_csv()
Excel               pd.read_ecel()      df.to_csv()
sql                 pd.read_sql()       df.to_csv()
###################################################
"""

#########################
""" PANDAS DATA TYPES """
#########################

"""
Below is a summary of data types in Pandas and their Python analogues

#################################################
Pandas Type                 Python Type
#################################################
object                      string
int64                       int
float64                     float
datetime64, timedelta[ns]   N/A (datetime module)
#################################################        
"""

# 'df.dtypes' tells us the data type

# 'df.describe()' outputs a summary of basic statistics of dataframe
# 'df.describe(include=all)' outputs all columns in statistics summary (incl. unique, top, freq.)

# 'df.info' gives top and bottom 30 rows of dataframe

###########################
""" ACCESSING DATABASES """
###########################

"""
How a user accesses a database with a code written on a Jupyter Notebook:
    User -> Jupyter Notebook -> API calls -> DBMS

Application Program Interface (API) is a set of functions that can be called to access
a service. Typical operation looks like this:

################################
        CONNECT ->          
        SEND ->     
API     EXECUTE ->          DBMS
        STATUS CHECK ->
        <- OK
        <- DISCONNECT
################################
"""

#####################
""" PYTHON DB-API """
#####################

"""
The Python DB-API is the Python standard API to access DBMS:
    User -> Jupyter Notebook -> DB-API calls -> DBMS

Two main concepts:

    Connection Objects:
        - Databases connections
        - Manage transactions
    Cursor Objects:
        - Database queries

Connections Methods:
    cursor()  # returns new cursor object using the connection
    commit()  # commit pending transactions to the database
    rollback()  # causes database to rollback to start of pending transactions
    close()  # use to close the connection
"""

###################
""" DP-API CODE """
###################

"""
Below is how we would use 'DB-API' code to establish a connection to a database.
"""

from dbmodule import connect

# create connection object
connection = connect("databasename", "username", "password")

# create a cursor object
cursor = connection.cursor()

# run queries
cursor.execute("select * from mytable")
results = cursor.fetchall()

# free resources
Cursor.close()
connection.close()

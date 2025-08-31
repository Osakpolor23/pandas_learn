# Hands_on pandas learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset to memory
df = pd.read_csv("diabetes.csv")

# Display the first 10 rows of the dataframe
print(df.head(n=10)) # default is 5 e.g print(df.head())

# Display summary statistics of the dataframe
print(df.describe())

# Display summary statistics of the dataframe with specific percentiles
print(df.describe(percentiles=[0.3,0.5,0.7])) # 30th, 50th, and 70th percentiles

# Display summary statistics of certain datatypes
print(df.describe(include=[int])) # only integer columns

# Display summary statistics by omitting certain datatypes
print(df.describe(exclude=[int])) # omit integer columns

# Display summary statistics with the dataframe transposed
print(df.describe().T) # Using the .T attribute to transpose the dataframe

# Understand the data using the .info
print(df.info(show_counts=True, memory_usage=True, verbose=True)) # show non-null counts, memory usage, and all columns info

# get the number of rows and columns
print(df.shape) # (rows, columns)
print(f"Number of rows: {df.shape[0]}") # number of rows
print(f"Number of columns: {df.shape[1]}") # number of columns

# get the column names
print(df.columns) # Index of column names
print(list(df.columns)) # column names as a list

# checking for missing values
# As the dataset has no missing values, let's make a copy of the dataframe and introduce NaN values
# which denote missing values, in rows 2 to 5 of the pregnancies column
df2 = df.copy() # using the .copy method
df2.loc[2:5, "Pregnancies"] = None # introducing NaN values
print(df2.head(7)) 

# checking for missing values using isnull() method in the first 7 rows
print(df2.isnull().head(7))
# prefferable to count the number of null values
print(df2.isnull().sum()) # sum of null values in each column
print(df2.isnull().sum().sum()) # total number of null values in the dataframe

# Sorting, Slicing and Extracting Data in Pandas
sorted_df = df.sort_values(by="Age", ascending=False) # sorting the dataframe by Age in descending order
print(sorted_df) # using the inplace keyword changes the original dataframe and trying to wrap in print returns None.

# sorted_df = df.sort_values(by=["Age", "Glucose"], ascending=[False, True]) # sorting the dataframe by Age in descending order and Glucose in ascending order
print(sorted_df) # we can as well keep the inplace keyword and then go ahead and print df afterwards, prints the changed original df without having to assign

# resetting the index as sorting might misalign the original index
df.reset_index(drop=True, inplace=True) # drop=True to avoid adding the old index as a column
print(df) # print the reset dataframe

# sorted_df.reset_index(drop=True, inplace=True) # reset the index of the sorted dataframe
print(sorted_df) # print the reset sorted dataframe

# filtering data using conditions
print(df[df["Glucose"] > 100]) # Selects rows where Glucose > 100

# isolating one or more specific column using square brackets
print(df["Outcome"]) # Selects the Outcome column -- a series which is a one-dimensional array. Multiple series makes a dataframe.
print(df[["Pregnancies", "Outcome"]]) # Selects the Pregnancies and Outcome columns

# isolating a row in pandas
print(df[df.index == 1])

# isolating multiple rows using the .isin() method
print(df[df.index.isin(range(2,10))]) # Selects rows with index 2 to 9

# using the .loc[] and .iloc[] to fetch rows
# .loc[] uses the label of the rows and columns
# .iloc[] uses the integer index of the rows and columns
df2 = df.copy() # create a copy of the original dataframe
df2.index = range(1,769) # modifying the index to start from 1
df2.loc[2:5, "Pregnancies"] = None
print(df2)
print(df2.loc[1]) # selects the row with index labelled 1
print(df2.iloc[1]) # selects the row with integer index 1

# fetching multiple rows using .loc[] and .iloc[]
print(df2.loc[100:110])
print(df2.iloc[100:110]) 

# subsetting rows with .loc[] and .iloc[] using lists instead of range
print(df2.loc[[100,200,300]]) 
print(df2.iloc[[100,200,300]])

# selecting rows and columns with .loc[] and .iloc[] -- The difference here is clear
print(df2.loc[100:110, ["Pregnancies", "Glucose", "BloodPressure"]]) # specifying the column labels
print(df2.iloc[100:110, :3]) # selecting the first three columns using slicing i.e. integer index 0, 1, 2

# For faster workflow, we can pass in the starting index of a row as a range e.g
print(df2.loc[760:, ["Pregnancies", "Glucose", "BloodPressure"]])
print(df2.iloc[760:, :3])

# updating and modifying values using the assignment operator
df2.loc[df2["Age"]==81, ["Age"]] = 80 # changing the Age value of 81 to 80
print(df2.loc[df2["Age"]==80, ["Age"]]) # verifying the change by selecting only the Age column

# conditional slicing(that fits certain conditions)
print(df[df.BloodPressure == 122]) # selecting rows where BloodPressure is 122

print(df[df.Outcome == 1]) # selecting rows where Outcome is 1
# slicing using comparison operator
print(df.loc[df["BloodPressure"] > 100, ["Pregnancies", "Glucose", "BloodPressure"]])

# Cleaning data using pandas
print(df2.isnull().sum()) # checking for missing values

# dealing with missing values 
# technique 1 -- dropping missing values
df3 = df2.copy() # create a copy of the dataframe with missing values
df3 = df3.dropna() # drop rows with any missing values
print(df3.shape) # returns (764, 9) which is 4 rows less than df2
print(df3.isnull().sum()) # verifying that there are no missing values

# The axis keyword lets you drop the column with missing values totalling by setting it to 1
df3 = df2.copy() # create a copy of the dataframe with missing values
df3.dropna(inplace=True, axis=1) # drops columns with any missing values
print(df3.head()) # Notice that the columns with missing values are dropped e.g Pregnancies

# we can as well drop all rows and columns with missing values using the how keyword e.g
df3 = df2.copy() # create a copy of the dataframe with missing values
df3.dropna(inplace=True, how="all") # drops all rows with any missing values

# dealing with missing values 
# technique 2 -- filling with mean values
df3 = df2.copy() # create a copy of the dataframe with missing values
# get the mean value of Pregnancies
mean_value = df3["Pregnancies"].mean() 
# fill the missing values using .fillna()
df3.fillna(mean_value, inplace=True)
print(df3.head(10)) # verifying that there are no missing values (in rows 2 to 5)

# dealing with duplicate data
# create duplicate data by using the .concat() method
df3 = pd.concat([df2, df2]) # concatenate the rows of df2 Dataframe to itself, creating duplicates.
print(df3.shape)  # outputs (1536, 9)

# remove duplicates using the .drop_duplicates() method
df3.drop_duplicates(inplace=True)
print(df3.shape)  # outputs (768, 9) which is the original shape of df2

# renaming columns using pandas
df3.rename(columns = {"DiabetesPedigreeFunction":"DPF"}, inplace=True)
print(df3.head())

# dealing with missing values
mean_value = df3["Pregnancies"].mean()
df3.fillna(mean_value, inplace=True)
print(df3.head())

# we can also directly assign column names as a list
df3.columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Outcome', 'STF']
print(df3.head())

# data analysis in pandas
# summary operator(mean, median, mode)
print(df3.mean()) # mean of all columns
print(df3.median()) # median of all columns
print(df3.mode()) # mode of all columns

# creating new columns based on existing columns
df2["Glucose_Insulin_Ratio"] = df2["Glucose"] / df2["Insulin"]  # creating a new column for the ratio of Glucose to Insulin
print(df2.head()) # displaying the new column

# working with categorical data
# we can count the number of observations in a column using the .value_counts() method
print(df["Outcome"].value_counts()) # counting the number of occurrences of each class 1 = diabetic, 0 = non-diabetic

# Adding the normalize argument returns proportions(like percentages) instead of absolute counts.
print(df["Outcome"].value_counts(normalize=True)) # returns the proportions of each class

# we can apply .value_counts() to specific columns as well by using the subset argument
print(df.value_counts(subset=["Pregnancies", "Outcome"])) # counts occurrences of each combination of Pregnancies and Outcome

# aggregating data with .groupby() method in pandas
print(df.groupby("Outcome").mean()) # mean of all columns grouped by Outcome

# aggregating using .groupby() with more than one columns
print(df.groupby(["Pregnancies", "Outcome"]).mean()) # mean of all columns grouped by Pregnancies and Outcome

# # aggregating data by pivoting with pandas
# we can calculate summary statistics and draw conclusions using a combination of variables
pivot_table_df = pd.pivot_table(df, values = "BMI", index = "Pregnancies",
                         columns = ["Outcome"], aggfunc=np.mean) # Select rows as unique values of Pregnancies, columns as
# unique values of Outcome, and the cells as the average BMI values for these categories
print(pivot_table_df) # e.g for pregnancy 5 and Outcome 0, the average BMI is 31.1

# visualization using pandas
# pandas provide convenience wrappers to Matplotlib plotting functions to make it easy to visualize dataframes e.g
# df[["BMI", "Glucose"]].plot.line() # line plot of BMI and Glucose vs the row index
# plt.show() # This opens up/displays the plot window -- not needed in Jupyter notebook as it is interactive

# we can set the colors ourself e.g
df[["BMI", "Glucose"]].plot.line(figsize = (20, 10),
                                 color = {"BMI":"red", "Glucose":"blue"})
plt.show()

# all the columns of the df can also be printed by using the subplots argument e.g
df.plot.line(subplots = True)
plt.show()

# for discrete columns such as Outcome with 0 and 1, we can use a bar plot over the category counts to visualize the distribution
df["Outcome"].value_counts().plot.bar()
plt.show()

# box plots in pandas
# we can plot the quartile distributions of a column in a boxplot in pandas
df.boxplot(column = ["BMI"], by = "Outcome") # boxplot of BMI grouped by Outcome
plt.show()
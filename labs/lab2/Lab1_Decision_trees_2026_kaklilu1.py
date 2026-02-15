#!/usr/bin/env python
# coding: utf-8

# # Lab 1: Decision trees - The CART algorithm
# 
# In this lab, you will be programming your own decision tree CART algorithm with Gini criterion.
# 
# In the following lines, complete the functions replacing the comments # YOUR CODE GOES HERE or the variables equal to 'None'  with your code.
# 
# To evaluate the algorithm we will be working with the [
# Statlog (Heart) Data Set](https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29) containing predictor variables about patients with Heart Disease.
# 

# **0. Moving the lab to your folder**
# 
# Copy this notebook to your personal directory as you don't have the permission to edit this file.
# 
# Go to "File" - "Save a copy in Drive"
# 
# Copy this notebook into your own drive and suffix your JHED ID (It's the part before the @ symbol in your email, not your Hopkins ID in the SIS)
# 
# * e.g. Lab1_Decision_tree_myjhedID
# 
# 

# In[2]:


# Library import. Feel free to add more libraries that you may use in your code
import urllib.request
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from pandas.api.types import CategoricalDtype


# **Data Set Information:**
# 
# https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
# 
# Attribute Information:
# 
# 1. age
# 2. sex
# 3. chest pain type (4 values)
# 4. resting blood pressure
# 5. serum cholesterol in mg/dl
# 6. fasting blood sugar > 120 mg/dl
# 7. resting electrocardiographic results (values 0,1,2)
# 8. maximum heart rate achieved
# 9. exercise induced angina
# 10. oldpeak = ST depression induced by exercise relative to rest
# 11. the slope of the peak exercise ST segment
# 12. number of major vessels (0-3) colored by flourosopy
# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# 
# Attributes types:
# 
# * Continuous: 1, 4, 5, 8, 10, 12
# * Binary: 2, 6, 9
# * Categorical: 3, 7, 13
# * Integer: 11
# 
# Variable to be predicted:
# 
# Absence (1) or presence (2) of heart disease

# #1. Data Loading

# In[3]:

if __name__ == "__main__":
  sData='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'
  urllib.request.urlretrieve(sData,'./datafile.txt') # The data is stored in your drive folder under the name 'datafile.csv'
  df=pd.read_csv('./datafile.txt', sep=' ')


# If you print the head of the dataframe, you can observe the different predictor variables and the labels of the first five observations

# In[4]:


  df.head()


# As you can see above, this dataset does not contain headers for each column, we can add it manually when loading the dataset based on the dataset information.

# In[5]:


# Reload the dataset and adding the header

# ISSUE: In machine learning it is a good practice to have a naming convention that doesn't use spaces, so that columns can be accessed as a variable like df.age instead of df['age']. So I will replace the spaces with underscores in the column names.
  df=pd.read_csv('./datafile.txt', sep=' ', header=None, names=['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholesterol', 'fasting blood sugar', 'resting electrocardiographic results', 'maximum heart rate achieved', 'exercise induced angina', 'oldpeak', 'the slope of the peak exercise ST segment', 'number of major vessels', 'thal', 'class'])

# It is a good practice to have a naming convention that doesn't use spaces, so that columns can be accessed as a variable like df.age instead of df['age']. So I will replace the spaces with underscores in the column names.
  df.columns = df.columns.str.replace(' ', '_')


# In[6]:


  df.head()


# In[7]:


# Using a set to get find out the unique values in the column 'the slope of the peak exercise ST segment'
  df['the_slope_of_the_peak_exercise_ST_segment'].unique().tolist()


# In the next cell, you can print a single observation by selecting its correspondent series using iloc.

# In[8]:


  series=df.iloc[3,:] # The last column is removed (it's the class)
  print(series)


# In[9]:


# FIX: Change to the markdown file here.


# **Dataset descriptors**
# 
# **Data and notation used in the algorithm:**
# 
# * **X:** The training subset containing N observations
# * **Each observation, x_n:** A vector containing the predictor variables (size M) for each observation.
# * **y_n:** The label of x_n
# * **J:** Total number of classes
# * **N:** Total number of observations (number of training vectors)
# * **M:** Number of predictor variables (feature vector length)
# 
# ### **TASK 1:** In the next cell, calculate the N, M and J.

# In[10]:


#First steps: Determine the number of features, classes and observations
#(replace 'None' with your code) (5 POINTS)

# Since this variables are used multiple times, I strong type them to catch possible errors early
  from pandas.core.api import DataFrame

  y: DataFrame =df['class']
  X: DataFrame =df.drop(columns=['class'])
  N: int = len(df.index)
  M: int = len(df.columns) - 1
  J: int = len(y.unique())


# ISSUE: Use f-strings for better readability and performance instead of string concatenation with + operator.
  print(f'Number of observations: {N}')
  print(f'Number of predictor variables: {M}')
  print(f'Number of classes: {J}')


# In[11]:


  display(df.dtypes)


# **Variable type:**
# As you can see, some categorical variables, e.g., sex, are stored as float type. Please redefine them based on the dataset information.

# In[12]:


# Redefine the variables data type based on attribute info (5 points)

# NOTE: For other datasets I would just change this set
  set(categorical_cols := [
      'sex',
      'chest_pain_type',
      'fasting_blood_sugar',
      'resting_electrocardiographic_results',
      'exercise_induced_angina',
      'number_of_major_vessels',
      'thal'
  ])

  for col in categorical_cols:
      df[col] = df[col].astype('category')
      X[col] = X[col].astype('category')
      

# instead of converting the columns to an object I can use a set to check if the columns are categorical or not
  print('Categorical columns: ', categorical_cols)


# In[13]:


  print(df.dtypes)


# ###**TASK 2:** Split the data into, approx, 80% training and 20% testing randomly

# In[14]:


# Split the data into, approx, 80% training and 20% testing
#(replace 'None' with your code)  (5 POINTS)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)   

  dfTrain = pd.concat([X_train, y_train], axis=1)
  dfTest = pd.concat([X_test, y_test], axis=1)

  print("Training set:")
  display(dfTrain.head())
  print("Testing set:")
  display(dfTest.head())


# # 2. Decision Tree and The CART Algorithm
# *1.	Find each predictor’s best split:*
# 
# * a.	Determine if the predictor is categorical or numerical
# 
# 
# * b.	For each predictor, obtain unique values.
# 
# * c.	For each predictor: using the unique values, go through each value to examine each predictor’s possible split point
# 
# 
# * - i.	The best split point (threshold) is the one that maximizes the splitting criterion1 (impurity gain) the most.
# 
# * - ii.	In categorical predictors, we will have to consider a possible node for each category.
# 
# 
# *2.	Find the predictor that provides the best split:*
# 
# * a.	Select the predictor that maximizes the splitting criterion (the one that produces the highest impurity gain.
# 
# 
# *3.	Split the data X into two new nodes using the split point calculated in point 1 for the predictor selected in point 2.*
# 
# 
# *4.	Repeat point 1 for each branch if stopping rules are not satisfied:*
# 
# * a.	Stopping rules:
# * * i.	Maximum tree depth (maximum number of nodes in a row)
# * * ii.	All the remaining training observations in a resulting branch belong to a single class (maximum purity)
# * * iii.	All observations in a node have the same predictor’s values: it is not possible to split more.
# * * iv.	The node size is less than the minimum node size (number of observations in a node)
# 
# In the following cells, create the functions required for each point.
# 

# ###**TASK 3:** 1.a.  Determine if the predictor is categorical or numerical

# In[15]:


# 1.a Is the class numerical? (5 POINTS)

def is_numerical(df):
    """Test if values in a series are numeric and returns a vector with boolean results"""
    return df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x) and x.name != 'category')


# In[16]:

if __name__ == "__main__":
  is_numerical(dfTrain)


# ###**TASK 4:** 1.b.	For each predictor, obtain unique values (categories)

# In[17]:


  dfTrain.dtypes


# In[18]:


#1.b Function to obtain categories per predictor (5 POINTS)
def unique_values(df,column):
  """
  This function returns a list with all the possible categories present in a predictor variable
  Inputs: dataframe (df) and string containing the name of the predictor variable (column)
  """
  #YOUR CODE GOES HERE
  return df[column].unique().tolist()


# In[19]:


if __name__ == "__main__":
  categories=unique_values(df,'thal')
  print(categories)
  print(categories[0])


# ###**TASK 5:** 1.c.	For each predictor: using the unique values, go through each value to examine each predictor’s possible split point
# 
# 1. c. i. Search for the best split point (threshold) is the one that maximizes the splitting criterion1 (impurity gain) the most.

# In[20]:


#1.c.i Generate the gini function that calculates impurity  (15 POINTS)
def impurity_gini(df):
  """
  Calculates and returns the Gini impurity of the input dataframe (df)
  """
  # Calculates the impurity using the Gini index
  
  if (df["class"].empty):
    return 0.0
  
  # Get the counts of each class
  _, counts = np.unique(df['class'], return_counts=True)
  
  # Calculate Maximum Likelihood Estimate of probabilities
  probabilities = counts / counts.sum()
  gini_impurity = 1 - np.sum(probabilities ** 2)
  return gini_impurity
  


def impurity_gain(df, dfl, dfr):
  """
  Calculates and returns the Impurity Gain considering that df is used to
  calculate the initial impurity (or node t), and dfl and dfr to calculate the
  weights and impurities of node L and R, respectively
  """
  # This ought to be maximized in the ID3 algorithm
  impurity_t = impurity_gini(df)
  impurity_l = impurity_gini(dfl)
  impurity_r = impurity_gini(dfr)
  
  weight_l = len(dfl) / len(df)
  weight_r = len(dfr) / len(df)
  
  gain = impurity_t - (weight_l * impurity_l + weight_r * impurity_r)
  return gain

if  __name__ == "__main__":

# In[21]:


  impurity_gini(df)


# In[22]:


  impurity_gain(df,df, df)


# ###**TASK 6:** Search for the split point (or threshold) that provides the highest impurity gain in numerical predictor variables (for instance, 'Age' is a numerical variable).

# In[23]:


#1.c.i Numerical split analysis (10 POINTS)
def numerical_max_search(df, column):
  """
  FIX: This doc is not clear on what the parameter 'column' is supposed to be. 
  Is it the dependant variable "Y" or one of the columns in the dataframe "X"? 
  I will assume it's one of the columns in "X" since we are trying to find the best split point for a predictor variable.

  Returns the max impurity gain (ig_max) of the variable defined by 'column' in the df subset.
  It also returns the value of the split point (category_max) that yields the
  highest impurity gain for that variable.
  """
  # Tryout all possible split points and calculate the impurity gain for each of them.
  # Return the maximum gain and the corresponding split point.
  unique_values, counts = np.unique(df[column], return_counts=True)
  ig_max = -np.inf
  catagory_max = None
  
  for value in unique_values:
    dfl = df[df[column] <= value]
    dfr = df[df[column] > value]
    
    gain = impurity_gain(df, dfl, dfr)
    
    if gain > ig_max:
      ig_max = gain
      category_max = value


  assert category_max is not None, "No valid split point found."
  return ig_max, category_max


# In[24]:

if __name__ == "__main__":
  ig_max, category_max = numerical_max_search(df, 'age')
  df.loc[df['age'] <= category_max, 'age']


# ###**TASK 7:** 1.c.	For each predictor: using the unique values, go through each value to examine each predictor’s possible split point
# 
# 1. c. ii.	In categorical predictors, we will have to consider a possible node for each category. This means that we have to perform a search of best split for each category. In the dataset that we are using, each of the categorical variables have only two possible categories (Yes or No - Male or Female). However, a code that consider more than 2 categories would be desirable.

# In[25]:


# 1.c.ii Categorical split analysis (10 POINTS)
def categorical_max_search(df, column):
    """
    Returns the max impurity gain (ig_max) of all categories in the variable defined by column
    it also returns the name of the category that yields the highest impurity gain (category_max).
    """
    #YOUR CODE GOES HERE
    unique_values = df[column].unique().tolist()
    ig_max = -np.inf
    category_max = None 
    for value in unique_values:
        dfl = df[df[column] == value]
        dfr = df[df[column] != value]
        
        gain = impurity_gain(df, dfl, dfr)
        
        if gain > ig_max:
            ig_max = gain
            category_max = value


    assert category_max is not None, "No valid category found."
    return ig_max, category_max


# In[26]:

if __name__ == "__main__":
  categorical_max_search(df, 'thal')


# ###**TASK 8:** 2.	Using the previous functions, find the predictor that provides the best split:
# a.	Select the predictor that maximizes the splitting criterion (the one that produces the highest impurity gain).
# 

# In[27]:


# 2.a Find predictor that provides best split (10 POINTS)
def find_max_predictor(df):
  """
  This function finds the predictor variable (Max_predictor), category or threshold
  (Max_category) and their associated impurity gain (Max_ig) in the input dataset (df)
  using the previous functions
  """
  Max_ig = -np.inf
  Max_predictor = None
  Max_category = None
  
  for column in df.columns:
    if column == 'class':
      continue  # Skip the target variable
    
    if is_numeric_dtype(df[column]):
      ig, category = numerical_max_search(df, column)
    else:
      ig, category = categorical_max_search(df, column)
    
    if ig > Max_ig:
      Max_ig = ig
      Max_predictor = column
      Max_category = category

  return Max_ig, Max_predictor, Max_category


# In[28]:

if __name__ == "__main__":
  Max_ig, Max_predictor, Max_category=find_max_predictor(df)
  print(f'Maximum Impurity Gain: {Max_ig} obtained with the predictor {Max_predictor} and category {Max_category}')


# ###**TASK 9:** 3.	Split any input data X (a dataframe) into two new nodes using the split point and predictor selected obtained with find_max_predictor.

# In[29]:


# 3. Split data (15 POINTS)
def split_data(df):
    """
    Splits the input subset (df) into two smaller subsets (dfL and dfR) that define
    the split that provides the highest Impurity Gain. It also returns the predictor
    (Max_predictor) and category (Max_category) that favor that split. The code might
    be different for numerical and categorical predictor variables.
    """
    Max_ig, Max_predictor, Max_category = find_max_predictor(df)
    if is_numeric_dtype(df[Max_predictor]):
            # Numerical predictor
        dfL = df[df[Max_predictor] <= Max_category]
        dfR = df[df[Max_predictor] > Max_category]
    else:
        dfL = df[df[Max_predictor] == Max_category]
        dfR = df[df[Max_predictor] != Max_category]
        
    return dfL, dfR, Max_predictor, Max_category


# In[30]:

if __name__ == "__main__":
# This defines the first node of the tree for the df dataset
  dfL,dfR,predictor,treshold=split_data(df)
# This calculates the Impurity Gain when going from the initial dataset (df) to
# the left and right subsets
  impurity_gain(df,dfL,dfR)


# **Additional code:**
# The following function and classes serve to store information of the nodes and leaves. You do not need to code anything here. :)

# In[31]:


def classify_leaf(df):
  """
  Counts the number of observations per class (class probability) in the input df
  This function serves to create the classification info stored in the leaf.
  """

  classes=df.iloc[:,-1] # Labels in df
  classes_uni=classes.unique() # this variable contains the classes that exist in df
  result: dict ={Tclass: (sum(classes==Tclass)/len(df)) for Tclass in classes_uni} #proportion of observations per class
  return result

class tree_leaf:
    """
    This class will contain the results of the leaf (ex: Positive: 0.8, Negative:0.2)
    """

    def __init__(self, df):
        self.predictions = classify_leaf(df)

class tree_node:
    """
    This class will contain the predictor and threshold of the node and the two
    nested nodes (left and right)
    """

    def __init__(self,
                 predictor,
                 threshold,
                 left_branch,
                 right_branch):
        self.predictor = predictor
        self.threshold = threshold
        self.left_branch = left_branch
        self.right_branch = right_branch


# ###**TASK 10:** Stopping rules.
# These rules will serve to decide when to stop looking for a node and create a leaf. Conventional algorithms might have more stopping rules, but we will restrict our rules to only four conditions. You will have to code the functions that check these rules.
# 
# 4.a.	Stopping rules:
# 
# i.	Maximum tree depth (maximum number of nodes in a row) (this will be checked in the main function)
# 
# ii.	All the remaining training observations in a resulting branch belong to a single class (maximum purity)
# 
# iii.	All observations in a node have the same predictor’s values: it is not possible to split more.
# 
# iv.	The node size is less than the minimum node size (number of observations in a node)
# 

# In[32]:


# check stopping rules (10 POINTS)

#4.a.ii
def check_impurity(df):
  """
  returns TRUE if immpurity of the data in df is equal to 0.
  """

  impurity = impurity_gini(df)
  return np.isclose(impurity, 0.0, atol=1e-8)


#4.a.iii
def check_predictor(df):
  """
  Checks if all the observations have exactly the same variable vectors
  (same values for every single predictor variable)
  """

  #YOUR CODE GOES HERE
  predictors = df.iloc[:, :-1]
  for column in predictors.columns:
      if len(predictors[column].unique()) > 1:
          return False

  return True

#4.a.iv
def check_node_length(df, min_partition_length):
  """
  returns TRUE if the number of observations in df is less than min_partition_length
  """
  return len(df) < min_partition_length
    #YOUR CODE GOES HERE



# If all your functions have been coded as requested, the following functions should train and evaluate a decision tree.
# 
# ### **TASK 11:** Write the training function of your decision tree

# In[33]:


# train the tree  (10 POINTS)
def train_tree_mlma(df, tree_depth=0, min_partition_length=2, max_tree_depth=200):

  dfL,dfR,predictor,threshold=split_data(df)
  tree_depth+=1 # the tree depth in this path is extended
  print('Tree depth: ' +str(tree_depth))
  # tree_depth==max_tree_depth checks condition 4.a.i
  if (check_impurity(df) 
      or check_predictor(df) 
      or check_node_length(df, min_partition_length) 
      or tree_depth==max_tree_depth):
    return tree_leaf(df)
  
  # Left branch (true)
  left_node = train_tree_mlma(dfL, tree_depth, min_partition_length, max_tree_depth)

  # Right branch (false)
  right_node = train_tree_mlma(dfR, tree_depth, min_partition_length, max_tree_depth)


  return tree_node(predictor, threshold, left_node, right_node)


# In[34]:

if __name__ == "__main__":
  my_tree=train_tree_mlma(dfTrain)


# In[35]:


def print_tree_mlma(my_tree, jump=''):
  jump=jump+'-'
  if isinstance(my_tree, tree_leaf):
    print(jump+'Result: ')
    print(my_tree.predictions)
  else:
    if is_numeric_dtype(my_tree.threshold):
      print(jump + '> If ' + str(my_tree.predictor) + ' <= ' + str(my_tree.threshold))
      print_tree_mlma(my_tree.left_branch,jump)
      print(jump+'> Else: ')
      print_tree_mlma(my_tree.right_branch,jump)

    else:
      print(jump + '> If ' + str(my_tree.predictor) + ' is ' + str(my_tree.threshold))
      print_tree_mlma(my_tree.left_branch,jump)
      print(jump+'> Else: ')
      print_tree_mlma(my_tree.right_branch,jump)


# In[36]:

if __name__ == "__main__":  
  print_tree_mlma(my_tree)


# In[37]:


def classify_tree_mlma(my_tree, my_series):
  """
  Classifies the input series and returns the predictions
  """
  if isinstance(my_tree, tree_leaf):
    return my_tree.predictions
  else:
    if is_numeric_dtype(my_tree.threshold): # Numeric
      if my_series[my_tree.predictor] <= my_tree.threshold:
        return classify_tree_mlma(my_tree.left_branch,my_series)
      else:
        return classify_tree_mlma(my_tree.right_branch,my_series)
    else: #Categorical
      if my_series[my_tree.predictor] == my_tree.threshold:
        return classify_tree_mlma(my_tree.left_branch,my_series)
      else:
        return classify_tree_mlma(my_tree.right_branch,my_series)


# In[38]:

if __name__ == "__main__":
  my_series=dfTest.iloc[1,:]
  prediction=classify_tree_mlma(my_tree, my_series)
  print('True class: ' + str(float(my_series.iloc[-1])))
  print('Predicted class: ')
  print(prediction.values())


# Code a function that calculates the accuracy of the trained tree when classifying the whole dfTest subset.

# In[39]:


# YOUR CODE GOES HERE (10 POINTS)
def get_accuracy(my_tree, dfTest):
    correct_predictions = 0
    total_predictions = len(dfTest)

    for index, row in dfTest.iterrows():
        true_class = row['class']
        prediction = classify_tree_mlma(my_tree, row)
        predicted_class = max(prediction, key=prediction.get)  # Get class with highest probability

        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy
if __name__ == "__main__":
  accuracy = get_accuracy(my_tree, dfTest)
  print(f'Accuracy on test set: {accuracy * 100:.2f}%')

# TODO: Answer Markdown


# Considering the size of the dataset we used in this lab, do you think the way we split this dataset into training and test dataset is a good choice？Analyze whether the dataset we used in this lab requires any specific validation in a real experiment or if the split we have done in this lab was enough. (5 points)

# **Answer:**
# 
# No, it definitely is not. Since we only had 270 observations, we should have made the most of the data available. We could have used cross-validation and hyperparameter grid search tuning to improve how well the decision tree fits the data. Because the dataset is so small, there is high variance in how the train–test split is determined. For example, simply changing the random state can lead to a 5–10 percent change in accuracy.

# Considering the data set we are using and the problem we are trying to solve, what is the benefit of using a decision tree over other types of classification approaches? Provide a detailed explanation to support your answer. (5 points)

# **Answer:**
# 
# The benefit of using a decision tree is that it is easily interpretable. We can use print statements or powerful tools like NetworkX to visualize the structure of the decision tree. Additionally, because a decision tree naturally mirrors how a doctor diagnoses patients through a series of yes-or-no questions, the model is intuitive and easy to understand.
# 
# Another advantage is that decision trees are classification algorithms that require very few assumptions about the data. In contrast, other classification algorithms such as QDA, LDA, and Naive Bayes require assumptions about the underlying data distribution or feature independence. With decision trees, we can simply provide the data and allow the CART algorithm to perform the classification.

# # 2. Random Forest

# ###**Task 12.1: build your own Random Forest (RF) algorithm**
# 
# In this task, you will implement a simple Random Forest-style ensemble using the decision tree training functions you developed earlier.
# 
# 1) **Training (Bootstrap Aggregation / Bagging)**  
# After splitting the data into training (80%) and test (20%) sets, train an ensemble of decision trees as follows:
# - For each tree, create a bootstrap sample by randomly sampling **with replacement** from the training set (use the same number of rows as the training set).
# - Train a decision tree on this bootstrap sample using your CART implementation from the previous tasks.
# - Repeat until you have trained `n_estimators` trees.
# 
# 2) **Prediction (Majority Vote)**  
# - For each test example, obtain predictions from all trees.
# - Combine the predictions by **majority vote** to produce the final predicted label.
# 
# 3) **Experiment and Comparison**  
# - Vary `n_estimators` and report how test accuracy changes as the number of trees increases.
# 

# In[40]:


# YOUR CODE GOES HERE (15 POINTS)

def my_tree_RF(df, tree_depth=0, min_partition_length=2, max_tree_depth=200):
    '''
    This function recursively builds a decision tree (a node) based on the given DataFrame df.
    It splits the data into left and right branches (dfL and dfR) based on some criteria.
    If certain stopping conditions are met (e.g., minimum node impurity, minimum partition length,
    or maximum tree depth), it assigns a leaf node.
    If not, it recursively calls itself to build the left and right branches.
    Finally, it returns the constructed tree node.
    '''
    return train_tree_mlma(df, tree_depth, min_partition_length, max_tree_depth)   


    # You should reuse and adapt your code from train_tree_mlma()


# In[41]:


def train_single_tree(dfTrain, dfTest):
    # Create a bootstrapped sample of the training data
    bootstrapped_df = dfTrain.sample(frac=1, replace=True)
    # Train a decision tree on the bootstrapped sample
    tree = my_tree_RF(bootstrapped_df)
  
    # get predictions for the test set
    predictions = []
    for index, row in dfTest.iterrows():
        prediction = classify_tree_mlma(tree, row)
        predicted_class = max(prediction, key=prediction.get)  # Get class with highest probability
        predictions.append(predicted_class)
    return predictions

def train_test(dfTrain, dfTest, n_estimator, n_jobs=-1):

  '''This function performs training and testing of the random forest model.
  It trains n_estimator decision trees using a bootstrapped sample from the
  training data and stores the predictions for each tree.
  It returns a list containing predictions made by each individual tree.'''


  if n_jobs == -1:
    n_jobs = os.cpu_count()  # Use all available CPU cores
    
  all_predictions = [None] * n_estimator 
  with ProcessPoolExecutor(max_workers=n_jobs) as executor:
      futures = {
        executor.submit(train_single_tree, dfTrain, dfTest): i 
        for i in range(n_estimator)
      }
      for future in as_completed(futures):
          i = futures[future]
          all_predictions[i] = future.result()
          print(f'Tree {i} completed.')
  
  # Transpose the list of predictions to have a list of predictions for each test instance
  # Row = test instance, Column = tree prediction
  return np.array(all_predictions).T
          
# YOUR CODE GOES HERE


# In[42]:


def evaluate_trees(dfTest, lists_preds, n_trees):

  '''This function evaluates the performance of the random forest model
  by aggregating predictions from individual trees and comparing
  them to the ground truth labels in the test set.
  It computes the accuracy of the ensemble predictions compared to the ground truth.
  The accuracy is printed out along with the number of estimators (trees) used.'''

  votes = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=lists_preds)
  bool_array = votes == dfTest['class'].values
  accuracy = np.sum(bool_array) / len(dfTest)
  print(f'Accuracy with {n_trees} estimators: {accuracy * 100:.2f}%')
  return accuracy
  # YOUR CODE GOES HERE


# In[43]:

if __name__ == "__main__":

  list_preds_trees = train_test(dfTrain, dfTest, 3)
  evaluate_trees(dfTest, list_preds_trees, 3)


# In[44]:


# Get most frequent value per column using apply
  most_frequent = dfTest.apply(lambda col: col.mode()[0] if not col.mode().empty else col.iloc[0], axis=0)
  print(most_frequent)


# In[45]:


  list_preds_trees = train_test(dfTrain, dfTest, 5)
  evaluate_trees(dfTest, list_preds_trees, 5)


# ###**Task 12.2**: Write code  with the help of sklearn packages to implement random forest algorithm and calculate the accuracy. Play with the hyperparameters, e.g., n_estimators, max_depth and the number of random state, report and discuss the influence on accuracy. You need to try dicuss the influence of at least two hyperparameters.

# In[52]:


  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
      
# YOUR CODE GOES HERE (5 POINTS)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  classifier = RandomForestClassifier(n_estimators=5, random_state=0)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f'Random Forest Classifier Accuracy with 5 estimators: {accuracy:.2%}')

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  classifier = RandomForestClassifier(n_estimators=30, random_state=0)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f'Random Forest Classifier Accuracy with 30 estimators: {accuracy:.2%}')


# Report and discuss the influence of at least two hyperparameters on accuracy (5 points)

# **Answer:**
# 

# Now compare the accuracy of your implementation to sklearn’s `RandomForestClassifier` on the same train/test split. Discuss which one is better and why. (5 points)
# 
# When experimenting with the random_state, I observed that the accuracy fluctuated significantly, changing by approximately 5% to 10% on average. This high variance is likely due to the small size of the dataset (only 270 observations), where different random splits for bootstrapping can drastically alter which difficult examples are included in training versus testing. In contrast, changing the max_depth (tree depth) had little to no effect on the final accuracy. This suggests that the Random Forest ensemble is robust enough to handle fully grown trees without overfitting, or that the critical decision boundaries are simple enough to be captured by shallower trees.
# 
# As shown with the print statement changing the number of estimators also tends to increase the accuracy up to a certain point (approximately 50)

# **Answer:**
# 
# In this experiment, my custom Random Forest implementation achieved an accuracy of 72.22%, while the Scikit-Learn RandomForestClassifier achieved 66.67% (using n_estimators=5 and random_state=0 for both).
# 
# The difference in accuracy is likely due to the random state initialization, which dictates exactly which data points are selected for each tree's bootstrap sample. With a small dataset and only 5 trees, the specific random samples chosen by your initialization happened to produce a better-performing model than Scikit-Learn's internal random selection just by chance. Additionally, Scikit-Learn initializes its trees to check only a random subset of features at each split, whereas your implementation checks all features, which can perform better when there are only a few strong predictors.

# ### Bonus Task: Boosting (AdaBoost) (10 points)
# 
# Implement a boosting ensemble using your decision tree code.
# 
# Training requirements:
# 1) Set sample weights w_i = 1/N on the training set.
# 2) For m = 1..n_estimators:
#    - Draw a weighted bootstrap sample from dfTrain using weights w (sample with replacement).
#    - Train a very shallow tree by setting max_tree_depth = 1 (or 2).
#    - Predict on the full training set and compute weighted error. If err >= 0.5, stop early (weak learner not useful).
#    - Compute alpha = 0.5 * ln((1 - err) / err).
#    - Update weights:
#        w_i <- w_i * exp(+alpha) if wrong else w_i * exp(-alpha), then renormalize.
# 
# Prediction:
# - For each test example, let each tree vote for a class with weight alpha.
# - Output the class with the larger total alpha vote.
# 
# Report test accuracy for several n_estimators and compare to your random forest.

# In[47]:


# YOUR CODE GOES HERE
from numpy import argmax


def adaptive_boosting(dfTrain, dfTest, n_estimators):
    # Train an ensemble of decision trees using the AdaBoost algorithm.

    # First, we initialize the weights for each training instance
    n_samples = len(dfTrain)
    weights = np.ones(n_samples) / n_samples    
    
    ensemble_of_trees = []  # This will store the trained trees in the ensemble
    alpha_values = []      # This will store the weights of each tree in the ensemble
    
    # Then we itteratively train decision trees, updating the weights based on the errors of the previous tree
    for i in range(n_estimators):
        
        # Sample randomly from the training data with replacement, using the current weights to influence the sampling
        dfTrain_sampled = dfTrain.sample(n=n_samples, replace=True, weights=weights)
        # Train a decision tree using the current weights
        tree = my_tree_RF(dfTrain_sampled, max_tree_depth=2)
        
        # Get predictions for the training set
        predictions = []
        for index, row in dfTrain.iterrows():
            prediction = classify_tree_mlma(tree, row)
            predicted_class = max(prediction, key=prediction.get)  # Get class with highest probability
            predictions.append(predicted_class)
        
        # Calculate the error of the current tree
        errors = (predictions != dfTrain['class']).astype(float)
        
        error_rate = np.sum(weights * errors) / np.sum(weights)
        
        if error_rate >= 0.5:
            break  # Stop this tree if its error rate is too high
            
        alpha_values.append(0.5 * np.log((1 - error_rate) / error_rate))
        ensemble_of_trees.append(tree)
        
        # update weights
        alpha = alpha_values[-1]
        weights[errors == 1] *= np.exp(+alpha)
        weights[errors == 0] *= np.exp(-alpha)
        weights /= np.sum(weights)

        
    # PREDICTIONS ON TEST SET

    # list of predictions
    test_predictions = []
    for _, row in dfTest.iterrows():
        class_scores = defaultdict(float)
        
        for tree, alpha in zip(ensemble_of_trees, alpha_values):
            prediction = classify_tree_mlma(tree, row)
            prediction_class = max(prediction, key=prediction.get)  # Get class with highest probability
            class_scores[prediction_class] += alpha
        
        final_prediction = max(class_scores, key=class_scores.get)  # Get class with highest aggregated score   
        test_predictions.append(final_prediction)
        
    return np.array(test_predictions)


# In[48]:

if __name__ == "__main__":
  dF_train, dF_test = train_test_split(df, test_size=0.2, random_state=0)
  predictions = adaptive_boosting(dF_train, dF_test, n_estimators=10)
  calculate_accuracy = np.sum(predictions == dF_test['class'].values) / len(dF_test)
  print(f'Accuracy of AdaBoost with 10 estimators: {calculate_accuracy * 100:.2f}%')


# In[49]:

if __name__ == "__main__":
  predictions


# Your colab notebook link:

# **You are ready to submit in Gradescope!**
# 
# Please suffix your colab file with _\<JHED ID\> (It's the part before the @ symbol in your email)
# 
# e.g. Lab1_Decision_trees_myjhedID
# 
# 4 easy steps to submit your lab:
# 
# 1.   Click on "Share" option on top right - Click on "copy link" option. Make sure your permission is set to "Anyone on the internet with this link can view". And paste it in the cell above.
# 2.   Go to "File" - "Download .ipynb" and "Download .py".
# 3.   Export the notebook to a PDF file with all the outputs.
# 3.   Upload the three files (.pdf, .ipynb, .py) to Gradescope.
# 
# That's it!

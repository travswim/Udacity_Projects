
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project: Building a Student Intervention System

# Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ### Question 1 - Classification vs. Regression
# *Your goal for this project is to identify students who might need early intervention before they fail to graduate. Which type of supervised learning problem is this, classification or regression? Why?*

# **Answer: The type of suerpervised learning for the student intervention problem is classification, as we are trying to use descretely classify which students need help. Regression is used for predicting continuous values**

# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.

# In[8]:

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score


# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"


# ### Implementation: Data Exploration
# Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
# - The total number of students, `n_students`.
# - The total number of features for each student, `n_features`.
# - The number of those students who passed, `n_passed`.
# - The number of those students who failed, `n_failed`.
# - The graduation rate of the class, `grad_rate`, in percent (%).
# 

# In[9]:

# TODO: Calculate number of students
n_students = len(student_data['school'])

# TODO: Calculate number of features
n_features = len(student_data.columns[:-1])

# TODO: Calculate passing students
n_passed = np.sum(student_data['passed'].str.count('yes'))

# TODO: Calculate failing students
n_failed = n_students-n_passed

# TODO: Calculate graduation rate
grad_rate = float(n_passed)/n_students*100

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# ## Preparing the Data
# In this section, we will prepare the data for modeling, training and testing.
# 
# ### Identify feature and target columns
# It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.
# 
# Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.

# In[10]:

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


# ### Preprocess Feature Columns
# 
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.
# 
# Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
# 
# These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.

# In[11]:

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))


# ### Implementation: Training and Testing Data Split
# So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
# - Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
#   - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
#   - Set a `random_state` for the function(s) you use, if provided.
#   - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.

# In[12]:

# TODO: Import any additional functionality you may need here
from sklearn import cross_validation

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, stratify=y_all, 
                                                    test_size=0.24, random_state=42)
print "Train set 'yes' pct = {:.2f}%".format(100 * (y_train == 'yes').mean())
print "Test  set 'yes' pct = {:.2f}%".format(100 * (y_test == 'yes').mean())
# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# ## Training and Evaluating Models
# In this section, you will choose 3 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.
# 
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Question 2 - Model Application
# *List three supervised learning models that are appropriate for this problem. For each model chosen*
# - Describe one real-world application in industry where the model can be applied. *(You may need to do a small bit of research for this — give references!)* 
# - What are the strengths of the model; when does it perform well? 
# - What are the weaknesses of the model; when does it perform poorly?
# - What makes this model a good candidate for the problem, given what you know about the data?

# **Answer: Using the Udacity lessons [1], intuition, and the scikit learn cheat-sheet [2], from the list of models above, it would appear that Naive Bayes, Support Vector Machine (SVM), and logistic regression would be best suited for solving the problem of student intervention. Reasons for choosing these models include the fact that there is a limited number of data points ( < 400), and the data is somewhat skewed. The graduation rate comprises of roughly 2/3rd of the data set. There are also 31 features for each data point, which can lead to issues relating to the curse of dimensionality [3]. 
# 
# Naïve Bayes is commonly used for spam filtering, with the benefit of working well with small data sets. However, a disadvantage of the Naïve Bayes model is the assumption of independent attributes. Given that Naïve Bayes performs well on small datasets, this model will likely perform well with the give dataset.
# 
# SVM is used in text and hypertext classification, or converting handwriting to text [4]. Benefits of SVM include the ability to work with small datasets with large dimensions. The kernel can be optimized to provide better results. It does not perform well with large datasets with lots of noise and overlapping classifications. The reason for choosing this model includes the fact that the dataset has a large dimension.
# 
# Logistic Regression is commonly used in medical applications, for instance diagnosing malignant tumors, and determining mechanical failure for engines [5]. Benefits include not having to worry about features being correlated (unlike Naïve Bayes), and is able to take in new data if the more data is expected to be produced (more students will arrive and will need to be categorised). Downfalls of logistic regression include requiring larger datasets to make results meaningful. This model may be useful, give that many of the features may be correlated, and logistic regression can handle this appropriately.
# 
# Works Cited
# 
# [1] 
# Udacity, "Machine Learning Nano-degree," 2016. [Online]. Available: https://www.udacity.com/. [Accessed 2016].
# [2] 
# scikit-Learn, "Choosing the right estimator," 2016. [Online]. Available: http://scikit-learn.org/stable/tutorial/machine_learning_map/. [Accessed 28 11 2016].
# [3] 
# UCLA, "Curse of Dimensionality," UCLA, 19 09 2001. [Online]. Available: http://www.stat.ucla.edu/~sabatti/statarray/textr/node5.html. [Accessed 28 11 2016].
# [4] 
# C. d. Souza, "Handwriting Recognition Revisited: Kernel Support Vector Machines," Code Project, 12 12 2014. [Online]. Available: https://www.codeproject.com/articles/106583/handwriting-recognition-revisited-kernel-support-v. [Accessed 29 11 2016].
# [5] 
# Coursera, "Machine Learning," Coursera, [Online]. Available: https://www.coursera.org/learn/machine-learning.
#  **

# ### Setup
# Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
# - `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
# - `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
# - `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
#  - This function will report the F<sub>1</sub> score for both the training and testing data separately.

# In[13]:

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))


# ### Implementation: Model Performance Metrics
# With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
#  - Use a `random_state` for each model you use, if provided.
#  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Create the different training set sizes to be used to train each model.
#  - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
# - Fit each model with each training set size and make predictions on the test set (9 in total).  
# **Note:** Three tables are provided after the following code cell which can be used to store your results.

# In[15]:

# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = SVC(random_state = 30)
clf_C = LogisticRegression(random_state = 30)

# TODO: Set up the training set sizes
X_train_100 = X_train.iloc[:100, :]
y_train_100 = y_train.iloc[:100]

X_train_200 = X_train.iloc[:200, :]
y_train_200 = y_train.iloc[:200]

X_train_300 = X_train.iloc[:300, :]
y_train_300 = y_train.iloc[:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)

for clf in [clf_A, clf_B, clf_C]:
    print "\n{}: \n".format(clf.__class__.__name__)
    for n in [100, 200, 300]:
        train_predict(clf, X_train[:n], y_train[:n], X_test, y_test)


# ### Tabular Results
# Edit the cell below to see how a table can be designed in [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). You can record your results from above in the tables provided.

# ** Classifer 1 - GaussianNB **  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |        0.0091           |     0.0007             |  0.7752          |   0.6457        |
# | 200               |        0.0009           |     0.0003             |  0.8060          |   0.7218        |
# | 300               |        0.0018           |     0.0005             |  0.8134          |   0.7761        |
# 
# ** Classifer 2 - SVC **  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |       0.0077            |     0.0013             |  0.8354          |   0.8025        |
# | 200               |       0.0052            |     0.0015             |  0.8431          |   0.8105        |
# | 300               |       0.0080            |     0.0026             |  0.8664          |   0.8052        |
# 
# ** Classifer 3 - Logistic Regression **  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |       0.1351            |     0.0003             |  0.8671          |   0.7068        |
# | 200               |       0.0026            |     0.0002             |  0.8211          |   0.7391        |
# | 300               |       0.0045            |     0.0004             |  0.8512          |   0.7500        |

# ## Choosing the Best Model
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score. 

# ### Question 3 - Choosing the Best Model
# *Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

# **Answer: Based on the results above, it can be seen that the SVC model produces a better F1 test score than both Logistic Regression and Naive Bayes. Logistic Regression is the most computationally efficient compared to both Naive Bayes and SVC. It should be noted that as more data is given to the models, the SVC computation time will grow exponentially and become more computationally expensive. Naive Bayes on the other hand will increase linearly with additional data. Logistic regression should be chosen based on having the ability to tune the model kernel to produce better results, and that its F1 test score is not that far off from SVC given its short prediction time. **

# ### Question 4 - Model in Layman's Terms
# *In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical or technical jargon, such as describing equations or discussing the algorithm implementation.*

# **Answer: Logistic Regression is commonly used for binary classification, i.e. scenarios where we have two possible outcomes such as wether or not a student will graduate. First, the data is split into a training and testing set (in this case 75%/25% training/testing). The features of the students (such as age, study time, internet availability etc) are then normalized to scale the features with large ranges, and then combined linearly using different weightings. The final result is fed through a sigmoid function which outputs a value between 0 and 1. Values >= 0.5 are flagged as students who need interventions, while values < 0.5 are not. We can then guage the accuracy of the model using various accuracy scores, and, if significantly accurate, implement student intervention methods to those students at risk of not graduating. Modifications to the model can be made to flag students to catch false negatives (students who will not graduate but are less likely to be caught by the model)**

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
# - Initialize the classifier you've chosen and store it in `clf`.
# - Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
#  - Set the `pos_label` parameter to the correct value!
# - Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.

# In[19]:

# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

# Normalize features
X_train = normalize(X_train)
X_test = normalize(X_test)

# Create the parameters list you wish to tune
#parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] }
parameters = {
    'C': np.logspace(-4,4,17),
    'penalty':['l1', 'l2'],
    'class_weight':[None, 'balanced'],
    }
scv = StratifiedShuffleSplit(y_train, test_size=0.25)

# Initialize the classifier
clf = LogisticRegression(random_state=30)

# Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
#grid_obj = GridSearchCV(clf, parameters, cv=scv, scoring=f1_scorer)
grid_obj = GridSearchCV(clf, parameters, cv=scv,
                        scoring=f1_scorer, verbose=1,
                        n_jobs=-1, pre_dispatch='2*n_jobs')

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
# print "F1 score for predicting all \"yes\" on test set: {:.4f}".format(
#     f1_score(y_test, ['yes']*len(y_test), pos_label='yes', average='binary'))
# print "\nF1 score for predicting all 'yes': {:.4f}".format(
#     f1_score(y_true = ['yes']*n_passed + ['no']*n_failed, y_pred = ['yes']*n_students, pos_label='yes', average='binary'))


# ### Question 5 - Final F<sub>1</sub> Score
# *What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

# **Answer: The final F1 score with tuning: **
# 
# Training: 0.8294
# Testing: 0.7891
# 
# **The F1 score without tuning: **
# Training: 0.8512
# Testing: 0.75
# 
# It can be seen that turning the model yielded an improvement in F1 scoring. After fiddeling a bit with the tuning parameters, it was apparent that changing the train/test split, and normalizing the features made a significant impact on the F1 scoring. Changing the random_state made little if no difference. **

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

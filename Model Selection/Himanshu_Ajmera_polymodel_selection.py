
# coding: utf-8

#                                                       # Module-10
# # Polynomial Feature Selection 

# In[1]:


#import packages
import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
import pandas as pd
get_ipython().magic('matplotlib inline')


# ### 1. Import Data

# In[2]:


#Read in data from a data file to data_df in DateFrame format

## Type your code here to import data from poly_data.csv to data_df
#########################################################
data_df=pd.read_csv('poly_data.csv')
#########################################################


#verify the dataframe is imported correctly 
print(data_df.head(6))


# ### 2. Observe Data

# In[3]:


#joint plot (or scatter plot) of X1 and y
sns.jointplot(data_df['X1'], data_df['y'])


# In[4]:


#joint plot (or scatter plot) of X2 and y
sns.jointplot(data_df['X2'], data_df['y'])


# In[5]:


#joint plot (or scatter plot) of X1 and X2
sns.jointplot(data_df['X1'], data_df['X2'])


# ### Based on observing the above 3 diagrams and the p-values displayed, we found both X1 and X2 have close correlation with y. X1 and X2 are independent from each other. 

# ### 3. Split the Data

# In[6]:


# split the data into training and testing datasets
# the percentage of training data is 75%

#split point 
percentage_for_training = 0.75

#number of training data 
number_of_training_data = int(len(data_df)*percentage_for_training)

#create training and testing datasets
train_df  = data_df[0:number_of_training_data]
test_df = data_df[number_of_training_data:]
print(train_df.shape)
print(test_df.shape)


# ### 4. Create Polynomial Features

# In[7]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#set the degree to 3, you can try a larger number if you like
#for degree = 3, we will generate 9 features. 
#open the link below to understand what these features are 
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html  
polynomial_features = PolynomialFeatures(degree=3)


# In[9]:


X_poly = polynomial_features.fit_transform(data_df[['X1','X2']])

#code to extract X for training and testing from the data frames

X_train = X_poly[0:number_of_training_data]
X_test = X_poly[number_of_training_data:]


# ### 5. Create and Train a Linear Regression Model

# In[10]:


# mse() calculates mean square error of a model on given X and y
def mse(X, y, model):
    return  ((y-model.predict(X))**2).sum()/y.shape[0]


# In[11]:


# use all the features to train the linear model 
lm = LinearRegression()
lm.fit(X_train, train_df['y'])
train_mse = mse(X_train, train_df['y'], lm)
print("Training Data Set's MSE is: \t", train_mse)
test_mse = mse(X_test, test_df['y'], lm)
print("Testing Data Set's MSE is : \t", test_mse)


# ## Reflection: 
# 1. What do the above MSEs mean? 
# 2. Is the current model with all the features good?
# 3. Should we try different feature sets? 
# 4. If yes, which one should we start with?  
# 

# ### 6. Use Lasso in Linear Regression to Penalize Large Number of Features

# In[12]:


#import lasso
#lasso is controlled by a parameter alpha.
#by fine tuning this parameter, we can control the number of features

from sklearn.linear_model import Lasso
#Train the model, try different alpha values.
Lasso_model = Lasso(alpha=0.15,normalize=True, max_iter=1e5, )
Lasso_model.fit(X_train, train_df['y'])
        


# In[13]:


#see the trained parameters. Zero means the feature can be removed from the model
Lasso_model.coef_


# In[14]:


#let's see the train_mse and test_mse from Lasso when 
#alpha = 0.15

train_mse = mse(X_train, train_df['y'], Lasso_model)
print("Training Data Set's MSE is: \t", train_mse)
test_mse = mse(X_test, test_df['y'], Lasso_model)
print("Testing Data Set's MSE is : \t", test_mse)


# In[15]:


#let's try a large range of values for alpha first
#create 50 alphas from 100 to 0.00001 in logspace
alphas = np.logspace(2, -5, base=10, num=50)
alphas


# In[16]:


#use arrays to keep track of the MSE of each alpha used. 
train_mse_array =[]
test_mse_array=[]

#try each alpha
for alpha in alphas:
    
    #create Lasso model using alpha
    Lasso_model = Lasso(alpha=alpha,normalize=True, max_iter=1e5, )
    Lasso_model.fit(X_train, train_df['y'])
    
    #Calculate MSEs of train and test datasets 
    train_mse = mse(X_train, train_df['y'], Lasso_model)
    test_mse = mse(X_test, test_df['y'], Lasso_model)
    
    #add the MSEs to the arrays
    train_mse_array.append(train_mse)
    test_mse_array.append(test_mse)
    


# In[17]:


#plot the MSEs based on alpha values
#blue line is for training data
#red line is for the testing data
plt.plot(np.log10(alphas), train_mse_array)
plt.plot(np.log10(alphas), test_mse_array, color='r')


# ### There is something interesting between 0 and 1 in the above diagram. 0 mean 10^0=1 While 1 means 10^1 = 10  so, we will look closely within this range to find the optimal alpha value
# 

# In[27]:


alphas = np.linspace(1, 10, 1000)
print(alphas)


# In[39]:


# We can try a smaller search space now (a line space between 1 and 10)
alphas = np.linspace(1, 10, 1000)
train_mse_array =[]
test_mse_array=[]

prev_diff = 0
### Type your code here 
## find train and test MSEs based on the alphas
## create the diagram below in which the blue line is train_mse_array
## the red line is test_mse_array
## Also, print out the alpha where the lines meet and the correponding 
## train_mse and test_mse
#################################################################
for alpha in alphas:
    #create Lasso model using alpha
    Lasso_model = Lasso(alpha=alpha,normalize=True, max_iter=1e5, )
    Lasso_model.fit(X_train, train_df['y'])
    
    #Calculate MSEs of train and test datasets 
    train_mse = mse(X_train, train_df['y'], Lasso_model)
    test_mse = mse(X_test, test_df['y'], Lasso_model)
    diff = abs(train_mse - test_mse)
    if(diff<prev_diff):
        prev_diff = diff
        best_alpha = alpha
        best_train = train_mse
        best_test = test_mse
  
       
    #add the MSEs to the arrays
    train_mse_array.append(train_mse)
    test_mse_array.append(test_mse)
    
    
#print optimal alpha and corresponding MSE values
print("The optimal alpha is",best_alpha)  
print("Train MSE is",train_mse)
print("Test MSE is",test_mse)
    


###################################################################
#plot the MSEs based on alpha values
#blue line is for training data
#red line is for the testing data
plt.plot(alphas, train_mse_array)
plt.plot(alphas, test_mse_array, color='r')


# ### By observing a smaller range of alpha, we can clearly see how the MSEs change as we change the model and features. Use the diagram to explain the trends of the two lines and summarize what you learned so far. 

# In[1]:


## type your code here to describe the above diagram and what you learned 
## so far about feature and model selection ( about 200 words )
############################################################################
print("As we can see from the diagram, the red curve which shows how MSE for test data is decreasing for the value of alpha within small range of 1 to 10 while the MSE for train data is increasing. The best alpha is the point where these two curves sre intersecting or have difference minimum. It is observable that while range of alpha was large, there was a sharp point where the difference was begin to increase after being almost same (parallel). While looking closely that is in small range of alphas the difference is more visible and clear. Graphs really helps in understanding the values in visualized medium.")
#############################################################################


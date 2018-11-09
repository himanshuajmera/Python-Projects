
# coding: utf-8

#                                                      # Module-9
# # Sklearn                                                     

# In[1]:


#import the needed packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


# read in app_usage data into vpn_df
vpn_df=pd.DataFrame.from_csv("app_usage.csv",index_col=None)


# In[3]:


#oberve the data
#RemoteAccess is the target, we will build a model that 
#takes all other columns as X to predict remote VPN usage

#look at the shape of the dataframe
print(vpn_df.shape)

#look at the column names
print(vpn_df.columns)

#show the first 5 records
print(vpn_df.head(5))


# In[4]:


#Visualize data

#1. See the distribtuion of Remote VPN Access

fig, ax = plt.subplots(figsize=(7,6))
ax = plt.hist(vpn_df['RemoteAccess'])
plt.grid(True)
plt.axis([0, 1200, 0, 18])
plt.xlabel("VPN Access")
plt.ylabel("Numbers")


# In[5]:


#2. See the correlation heat map
corrmat = vpn_df.corr()
fig, ax = plt.subplots(figsize=(12,10))
ax = sns.heatmap(corrmat, vmin= -0.8, vmax=0.8, square = True, annot=True, cmap="RdBu_r")
fig.tight_layout()


# In[6]:


#We determine that this is supervised machine learning problem
#Linear Regression can be a good model
from sklearn import linear_model


# In[7]:


#The function takes X, y and retrun the trained model and R squared
def train_model(X,y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    R_2 = model.score(X,y)
    return model, R_2

#create a function to calculate Adjusted R_square
# n is the number of sample, p is the number features
def cal_adjusted_R(R_2, p, n):
    R_adjusted = R_2-(1-R_2)*(p/(n-p-1))
    return R_adjusted


# In[8]:


#R_2_array stores the R squared of all the features
R_2_array = np.array([])

#Calcuate the R_squared 
for col_name in vpn_df.columns:
    if col_name == 'RemoteAccess':
        continue
    else:
        #extract the feature column from dataset
        variate = vpn_df[[col_name]]
        
        #y is still the last column
        covariate = vpn_df[['RemoteAccess']]
        
        # call the train_model() function to calculate R squared 
        model, R_2 = train_model(variate,covariate)
        
        #print feature and corresponding R-Squared value
        print(col_name,":" , R_2)
        
        #Add R-Squared value to an array
        R_2_array = np.append(R_2_array,R_2)
        

################################################################
## sorted_R_2_index stores the index numbers of R_2_array   ####
## in descending order of the R_2 values                    ####        
sorted_R_2_index = np.argsort(R_2_array)[::-1]       
#################################################################


#print out the sorted indexes 
print("The order of index numbers are : \t", sorted_R_2_index)


# In[9]:


#gradually build up our model and add R squared and adjusted R to the output

for i in range(len(sorted_R_2_index)):
    
    #the selected_features should be the top i most associated features
    selected_features = []
    
    #take the top 1 to ith features as X
    for j in range(i+1):
        
            #append a new column based on the sorted R value
            #take your time to digist this line
            selected_features.append(vpn_df.columns[sorted_R_2_index[j]])
            
    #verify we got the right features
    print(selected_features)
    
    # X
    X_feature = vpn_df[selected_features]
    
    # y
    target = vpn_df[['RemoteAccess']]
    
    # train the model
    model, R_2 = train_model(X_feature, target)
    
    #calculate adjusted R
    R_adjusted = cal_adjusted_R(R_2, i+1, vpn_df.shape[0])
    
    #print the output
    print("R2: ", R_2, "\t Ajusted R2: ", R_adjusted, "\n")


# In[10]:


#let's build the model with all the features

y = vpn_df['RemoteAccess']
X = vpn_df.drop('RemoteAccess', 1)

from sklearn import linear_model

#create a linear regression model from linear_model package 
model=linear_model.LinearRegression()

#Train the model with our data (X, y)
model.fit(X,y)

#Display the parameters
print('Intercept: \n', model.intercept_)
print('Coefficients: \n', model.coef_)

#use R squared to see how much variation is explained by the trained model
print('R_squared: \n', model.score(X,y))


# In[11]:


from sklearn.feature_selection import VarianceThreshold


# In[42]:


# 1. after reading the above article, you decide to keep only one feature to represent 
# all the features that have correlation higher than 0.9 to it. 
##################################################################################
## Removing the features with correlation higher than 0.9 to it
X = vpn_df.drop(['RemoteAccess'], 1)

#################################################################################


# In[43]:


for i in range(len(X.columns)):
    X_new = X.iloc[:,i:i+3]
    print(X_new)


# In[44]:


# 2. we use Lasso to further penalize models with more features
from sklearn.linear_model import Lasso

# in Lasso, the score is still R squared 
best_score = 0
score = []
# Lasso has a parameter alpha used to adjust the level of penalizing the 
# number of features. A bigger alpha will produce less features. 
# We initiate the best alpha to 0 
best_alpha = 0 
alphas = []

X_modified = []
# let's fine tune alpha to find the model we need 
for alpha in np.linspace(1,0.2, 1000):
    
    #create a linear regression (Lasso) model from linear_model package 
    model=Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    for i in range(len(X.columns)):
        X_new = X.iloc[:,i:i+3]
        #Train the model with our data (X, y)
        model.fit(X_new,y)
        best_score = model.score(X_new,y)
        best_alpha = alpha
        score.append(best_score)
        alphas.append(best_alpha)

best_score = max(score)      #best_score will be the maximum value in the list of scores
print("The best R of my 3-feature model is:\t\t", best_score)
print("The alpha I used in Lasso to find my model is: \t", best_alpha)

    #use R squared to see how much variation is explained by the trained model
    #print('R_squared: \n', model.score(X,y))


# In[47]:


##### Write your summary here
print("My summary: To obtain the best fitted model I randomly picked 3 features at a time using for loop, before that I removed the target feature. Once I got the best R value for each fitted model with three features I considered that model as final and wrote corresponding value of alpha.")
print("the 3 features in my model are: ERP, CRM, CloudDrive")


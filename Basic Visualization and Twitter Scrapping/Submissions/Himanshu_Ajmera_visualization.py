
# coding: utf-8

# # Requirement-1

# In[87]:


#firstly, we will import all the required packages
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[235]:


#to get the dataframe, pandas clipboard function is used.
df=pd.read_clipboard()


# In[236]:


df


# In[237]:


#since this column doesn't contribute anywhere.
df=df.drop("Rank",1)


# In[328]:


#to convert the values into numerics
df['Math'] = pd.to_numeric(df['Math'])
df['Reading'] = pd.to_numeric(df['Reading'])
df['Science'] = pd.to_numeric(df['Science'])


# In[329]:


#defined another column which is average of all the columns in dataframe
df["Average"] = (df["Math"]+df["Science"]+df["Reading"])/3


# In[330]:


#to see first five or header of the table/dataframe
df.head(5)


# In[331]:


x=df["Average"]
y=df["Math"]


# In[332]:


import matplotlib.patches as mpatches


# In[333]:


#histogram plot of "Average" and "mathematics" score
plt.title("Assignment 6")

plt.xlim(360,630)


plt.xlabel('Value Range')


n=[360,390,420,450,480,510,540,570,600,630]
plt.xticks(n)


#creating legend
red_patch = mpatches.Patch(color='red', label='x distribution')
blue_patch = mpatches.Patch(color='blue', label='y distribution')
plt.legend(handles=[red_patch,blue_patch])


plt.grid(color='g', linestyle='dotted')
plt.hist(x, alpha=0.5, color="red")
plt.hist(y, alpha=0.5, color="blue")





# In[334]:


from scipy import stats 


# # Requirement-2

# In[364]:


#definition of outlier finction
def find_outlier(a):
    Mean = np.mean(df[a])
    STD = np.std(df[a]) 
    factor = (1.8*STD)+Mean
    for index in df[a]:
        if(index > factor):
            row = df[df[a] == index]
            newList = []
            newList.append(row["Country"])    #creating a list of outlier in a given column
            print(" The outlier in " + a + " are ") 
            print(newList)


# In[365]:


#for Math column
find_outlier("Math")


# In[366]:


#for Average column
find_outlier("Average")


# In[367]:


#for Reading column
find_outlier("Reading")


# In[368]:


#for Science column
find_outlier("Science")


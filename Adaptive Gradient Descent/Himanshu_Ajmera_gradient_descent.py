
# coding: utf-8

#                                   # Module-7
# # Notebook- Gradient Descent

# Find the optimum of f(x)=x4+200*(x+2000)2+1000 using gradient descent
# 

# In[1]:


#import python packages
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


#define the x's range for plotting
x=np.arange(-120, 80, 0.1)


# a. f(x)
# 

# In[3]:


#define f(x) based on the function 
def f(x):
    return (x**4)+(200*(x+2000)**2)+1000 
    


# In[4]:


#plot x vs. f(x)
plt.plot(x,f(x))


# ### We can see that the minimun is between f(-50) and f(-100)

# b. df(x) #derivative of f(x)

# In[5]:


#define the derivative of f(x) over x ---> f'(x)
def df(x):
    return 4*(x**3) + 400*(x+2000)    


# In[6]:


#define a function to find the minimum of f(x) 
def find_optimum(x_old,x_new,gamma,precisions):
    #store each update in x_search
    x_search = [x_new]
    while abs(x_new-x_old) > precision:
        x_old = x_new
        x_new = x_old - gamma*df(x_old)
        x_search.append(x_new)
        
    print(len(x_search),"iterations")
    print("The local minimum occurs at %f" % x_new)
    print("gamma =",gamma)
    plt.plot([f(x) for x in x_search])


# c. find_optimum(x_old,x_new,gamma,precisions)

# In[7]:


#Test the find_optimum() function with the following parameters

x_old = 70 
x_new = 50 
gamma = 0.000001
precision = 1e-12 
find_optimum(x_old,x_new,gamma,precision)  #call the function 


# In[10]:


#create a find_optimum function to automatically set gamma based on 
#t is the decrease rate of gamma

def adaptive_optimum(x_old, x_new, gamma, t, precision):
    
    #nextIter is the flag for continuing or stopping the loop
    nextIter = True
    
    #keep searching until nextIter is set to false
    while nextIter:
        
        #decrease the value of gamma in each iteration
        gamma *=t
        
        #create a local copy of x_old and x_new in each iteration
        #it's because we can want any change to x_new and x_old to 
        #affect the calculation in the next iternation 
        x_old_try = x_old 
        x_new_try = x_new 
        
        #try 10000 times to see if x converges
        for i in range(10000):
            #use x_old_try to keep the value of x before the update
            x_old_try = x_new_try          
            
            try:
                x_new_try = x_old_try - gamma * df(x_old_try)
                if abs(x_new_try - x_old_try) < precision:
                    nextIter = False
                    print("Found gamma :", gamma)
                    print("The minimum is at  :",x_new_try)
                    print("The minimum of f(x) is :" ,f(x_new_try))
                    return

            # if there is an error, such as "value too large" error, stop the
            # iternation and try next gamma
            except:                          
                break
                
            
    return


# In[11]:


#Idealy, we want to automatically find the right gamma
#read http://www.onmyphd.com/?p=gradient.descent
#use backtracking method and create t

x_old = 70 # This value does not matter 
x_new = 50 # This value does not matter either

#the precision is set to be very high
precision = 1e-6

#decrease rate of gamma
t=0.9

#we can start with a large positive gamma close to 1
gamma = 1

#call the function
adaptive_optimum(x_old, x_new, gamma, t, precision)


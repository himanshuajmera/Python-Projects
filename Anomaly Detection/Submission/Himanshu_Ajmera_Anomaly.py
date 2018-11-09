
# coding: utf-8

#                                 Module-5
# # Anamoly Detection Function

# In[24]:


#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt


# In[25]:


#importing provided dataset as an array
data_set = open("anomaly_detection.txt",'r').read()   
data_list = data_set.split()
for i in range(len(data_list)):
    data_list[i] = float(data_list[i])
data_array = np.array(data_list)


# In[26]:


#this is our dataset, stored in the array.
data_array


# In[27]:


#though not necessary but plotting can give us a better insight into the data
plt.hist(data_array)
plt.show()


# In[38]:


#definition of function

def anomaly_detection(x):
    for i in range(l):
        a_sl = np.delete(x,[i])
        num_ = x[i]
        me = np.mean(a_sl)
        sd = np.std(a_sl)
        sub = abs(num_ - me)
        mul = 3*sd
        with np.errstate(divide = 'ignore'):
            if sub > mul:
                num = x[i]
                print("Remove",num,"from the list because it is",round(sub/sd,2),"times standard deviation of the list without it")
                print(num,"is removed from thelist.")
                a_sl = np.delete(x,[i])
                length = len(a_sl)
                for n in range(length):
                    a_sl_ = np.delete(a_sl,[n])
                    me = np.mean(a_sl_)
                    sd = np.std(a_sl_)
                    sub1 = abs(a_sl[n] - me)
                    mul1 = 3* sd
                    if sub1 > mul1:
                        num1 = a_sl[n]
                        print("Remove",num1,"from the list because it is", round(sub1/sd,2),"times the standard deviation of the list without it.")
                        print(num1,"is removed from the list.")
                        result = np.delete(a_sl,[n])
                        length = len(result)
                        for r in range(length):
                            a_sl_result = np.delete(result,[r])
                            mean = np.mean(a_sl_result)
                            std = np.std(a_sl_result)
                            sub2 = abs(result[r]- mean)
                            mul2 = 3 * std
                            if sub2 > mul2:
                                output = result[r]
                                print("Remove",round(output,2),"from the list because it is",round(sub2/std,2),"times the standard deviation of the list without it.")
                                print(round(output,2),"is removed from the list.")
                                print("No more Anomaly detected!")
                                return()
                            result_output = anomaly_detection(a_ar)
                            a = np.delete(result,[r])
                            length = len(a)
                            for z in range(length):
                                a_sl_result_ = np.delete(a,[z])
                                mean_ = np.mean(a_sl_result_)
                                std_ = np.std(a_sl_result_)
                                sub2_ = abs(a[z] - mean)
                                mul2_ = 3 * std_
                                if sub2_ > mul2_:
                                    output_ = a[z]
                                    print("No more Anomaly detected!")
                                    return()
                               


# In[40]:


#calling function for output.
r = anamoly_detection(data_array)


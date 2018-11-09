
# coding: utf-8

#                                                     Module-3 
# ## Notebook-3

# b. Replace all the punctuation marks with a space. 

# In[1]:


import string
from collections import Counter

#b
str = open("Sense_and_Sensibility.txt",'r').read()
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
str1 = str.translate(translator)


# a. The words should not be case sensitive, meaning “Mother” and “mother” are considered
# the same word. 

# In[3]:


#a 
clean_text = str1.lower()


# c. Use the “stopwords.txt” file to remove all the stop words in text. (Do NOT modify the
# stopwords.txt file)
# 

# In[4]:


#c

another_list = []

stp = open("stopwords.txt",'r').read()
mylist = stp.split()

t = clean_text.split()

for x in t:
    if x not in mylist:           # comparing from the list and removing it
        another_list.append(x)
        


# d. Create a histogram similar to the “histogram.jpg” file. The diagram should contain the
# ranking, the top 30 words, the number of times they appeared in the book. The number of
# stars will be the number of appearance divided by 10. For example, “mother” appears 263
# times; there are 26 stars displayed. (You may not have the exactly the same result as in the
# histogram.jpg)

# In[7]:


from collections import Counter
counts = Counter(another_list)


# In[8]:


top_30_words = counts.most_common(30)


# In[9]:


top_30_words_list = list(top_30_words)


# In[46]:


#getting histogram
i=0

for k,v in top_30_words_list:
    if i>30:
        break
    print(i,':',k,"(",v,"times)",end="\t\t\t\t\t")
    
    to_print = int(v/10)
    for j in range(to_print):
        print("*",end="")
    print()    
    i+=1


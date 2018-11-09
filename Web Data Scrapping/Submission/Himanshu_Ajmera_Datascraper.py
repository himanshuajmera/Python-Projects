
# coding: utf-8

# In[1]:


get_ipython().system('pip install bs4')


# In[2]:


from bs4 import BeautifulSoup

import requests

url = 'https://www.symantec.com/security_response/landing/vulnerabilities.jsp'


# In[3]:


response = requests.get(url)
content = response.content


# In[4]:


get_ipython().system('pip install lxml')


# In[5]:


get_ipython().system(' pip install html5lib')


# In[6]:


soup = BeautifulSoup(content, 'html5lib')


# In[7]:


tables = soup.find("tbody")


# In[8]:


from datetime import datetime


# In[9]:


rows = tables.findAll('tr')


# In[10]:


vul_names = []
date_values = []
url_values = []

for tr in rows:
    a_tag =  tr.find('a')
    vul = a_tag(text=True)[0]
    vul_names.append(vul)
    
    url_values.append("http://www.symantec.com"+a_tag['href'])
    
    td = tr.findAll('td')
    date = str(td[-1]).replace("<td>",'').replace("</td>","")
    date_values.append(date)
    


# In[11]:


update_date_values = []

for d in range (len(date_values)):
    update_date_values.append(datetime.strptime(date_values[d],"%m/%d/%Y"))
    
    


# In[12]:


type(update_date_values[0])


# In[13]:


type(date_values)
print(url_values[30])


# In[14]:


print(date_values[1])


# In[16]:


user_word = input("Type the word or phrase you look for :")
ip_strt_date = input("Type a start time (format - 2015/8/15):")
ip_end_date = input("Type a end time (format - 2016/5/13):")
strt_date = datetime.strptime(ip_strt_date,"%Y/%m/%d")
end_date = datetime.strptime(ip_end_date,"%Y/%m/%d")

for index in range(len(vul_names)):
    if (strt_date < update_date_values[index] < end_date):
        if (vul_names[index].count(user_word)>0):
            print(vul_names[index])
        


# In[17]:


foundCount = 0
for index, vulnerability in enumerate(vul_names[:100]):
    if user_word in vulnerability:
        response = requests.get(url_values[index])
        content = response.content
        soup = BeautifulSoup(content, "html5lib")
        description = soup.find(text ='Description').findNext('p')
        descp = str(description).replace("<p>",'').replace("</p>",'')
        date = soup.find(text = 'Date Discovered').findNext('p')
        the_date = str(date).replace("<p>",'').replace("</p>",'')
        foundCount += 1
        print('Found vulnerability number ' + str(foundCount) + ' in row ' + str(index + 1) + ' < ' + the_date,' >')
        print(descp)
        print (url_values[index])
        print("----------------------------------------")


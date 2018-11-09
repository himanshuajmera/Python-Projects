
# coding: utf-8

# In[1]:


get_ipython().system('pip install twython')


# In[2]:


from twython import Twython


# In[3]:


App_key = 'XtYVWtBGdSVwC7b5kNwqEciXA'
App_secret = 'mZu6EWHdMlLPXoUrmyJQAtmxl6ypvSlfGPR11t8476Ru3P9rPj'
Oauth_token = '2465384346-eaNI56mOW2qSURCzKGXKNNfpp8o3obZWIyzfAEm'
Oauth_token_secret = 'OeBc6y2zB5wNOY8g3MXV2PntmwS7wsI5JjDdDbV330kJv'

twitter = Twython(App_key,App_secret,Oauth_token,Oauth_token_secret)


# In[4]:


twitter


# In[5]:


search = twitter.search(q = 'machine')


# In[11]:


type(search)


# In[6]:


print(search)


# In[15]:


print(search['statuses'][0]['text'])


# In[8]:


from twython import TwythonStreamer


# In[17]:


tweets=[]
class MyStreamer(TwythonStreamer):
    def on_success(self, data):
        if data['user']['lang'] == 'en':
            tweets.append(data)
            print("received tweet #", len(tweets))
        if len(tweets) >= 50:
            self.disconnect()
    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()


# In[18]:


#Streaming the tweets about the term 'Technology'
stream = MyStreamer(App_key,App_secret,Oauth_token,Oauth_token_secret)
stream.statuses.filter(track='Technology')


# In[90]:


#print all the tweets' content, I expected in well format.
tweet_list = []
for i in range (len(tweets)):
    tweet_list.append(tweets[i]['text'])
    
print(tweet_list)


# In[92]:


#saving as text file.
file = open("50-tweets.txt","w")
file.write(str(tweet_list))
file.close()


# In[34]:


#just curious about location, name and usernames
for i in range (len(tweets)):
    print(tweets[i]['user']['location'])


# In[36]:


for i in range (len(tweets)):
    print(tweets[i]['user']['name'])


# In[47]:


for i in range(len(tweets)):
    print(tweets[i]['user']['screen_name'])


# In[48]:


#Trying to find attributes from the user's details
tweets[0]['user']


# # So while playing with this simple command I got an idea to find a relation between no. of statues and follower counts. 

# In[56]:


#Creating the list of number of statuses
statuses = []
for i in range(len(tweets)):
    statuses.append(tweets[i]['user']['statuses_count'])
    


# In[54]:


#creating the list of number of followers
followers = []
for i in range(len(tweets)):
    followers.append(tweets[i]['user']['followers_count'])
    


# In[60]:


#creating a dataframe
import pandas as pd
df = pd.Series(list(followers),list(statuses))


# In[61]:


df


# In[62]:


#importing libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[74]:


#reshaping the array to get well fitted model
follo = np.reshape(followers,(-1,1))
stat = np.reshape(statuses,(-1,1))


# In[71]:


# Create linear regression object
regr = linear_model.LinearRegression()


# In[75]:


#fitting the model
regr.fit(follo, stat)


# In[78]:


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(follo, stat))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(follo, stat))


# In[77]:


#plotting the result to further visualize
plt.scatter(follo, stat, color='black')
plt.plot(follo, stat, color='blue', linewidth=2)

plt.xticks(())
plt.yticks(())

plt.show()

After fitting a linear regression model I found that there is no relation between number of tweets or statuses somebody post and followers they have. As the MSE value is very high. It's just a coincidence that if you are tweeting back to back you get more followers. May be your content is good, to further analyze we can get a text in the tweets and how many followers they have, just you have to read every tweet to declare it as quality content or not which can be subjective. Anyways I'm happy that my old belief that be active means more fan following has some analyzed proof that it was myth.
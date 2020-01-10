
# coding: utf-8

# In[32]:

# get_ipython().magic('matplotlib inline')

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

train = pd.read_csv('data/train.csv')


# In[4]:

test = pd.read_csv('data/test.csv')


# In[6]:

submission = pd.read_csv('data/submit_example.csv')


# In[3]:

# train.head()


# In[9]:

title = train['title']


# In[11]:

result = []
for x in title:
    result.append(list(jieba.cut(x)))


# In[15]:

r2 = []
for x in result:
    r2.append(' '.join(x))


# In[16]:

# len(r2)


# In[18]:

# r2[:2]


# In[19]:

tf = TfidfVectorizer()
train_x = tf.fit_transform(r2)


# In[22]:

model = lgb.LGBMClassifier()
model.fit(train_x, train['flag'])


# In[23]:

test_x = []
for x in test['title']:
    test_x.append(' '.join(list(jieba.cut(x))))


# In[24]:

test_x = tf.transform(test_x)


# In[25]:

pred = model.predict_proba(test_x)


# In[28]:

submission['score'] = pred[:,1]


# In[ ]:




# In[33]:

submission['score'].plot(kind='hist')


# In[35]:

train['flag'].value_counts(1)


# In[34]:

del submission['flag']


# In[39]:

submission['flag'] = np.where(submission['score']>0.4, 1, 0)


# In[40]:

submission['flag'].value_counts(1)


# In[42]:

submission[['id','flag']].to_csv('sub.csv', index=None)


# In[41]:

# submission.head()


# In[5]:

# test.head()


# In[7]:

# submission.head()


# In[ ]:




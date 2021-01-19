#!/usr/bin/env python
# coding: utf-8

# In[3]:


import ast
with open('etnews_link.txt') as f:
    news = f.read()

news_link = ast.literal_eval(news)


# In[4]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime,timedelta


# In[8]:


def getPastWeekNews(URL):
    '''
    Input: Name of the company ( as per the name in txt file)
    Returns pandas dataframe containing(name of company, date, title, body source)
    
    
    '''
    url = URL
#     print(url)
    frame = []
    upperframe = []
    r_i = requests.get(url)
    soup_i = BeautifulSoup(r_i.content, 'html.parser')
    home = 'https://economictimes.indiatimes.com'
    links = [home + i['href'] for i in soup_i.select('.eachStory a')]
    
    for j in links:
#         print(j)
        try:
            r1 = requests.get(j)
            soup1 = BeautifulSoup(r1.content, 'html.parser')
            s = soup1.select_one('.bylineBox time').text.replace('Last Updated:','').strip()
            date = datetime.strptime(s, '%b %d, %Y, %I:%M %p IST')
            date1 = date.strftime('%B %d, %Y / %I:%M %p IST')
            if date>=datetime.today()-timedelta(7):
#             print(date)
                title = soup1.select_one('h1.artTitle').text.strip()
    #             print(title)
                news = ''.join([i.text.strip() for i in soup1.select('.artText')])
                frame.append([ date1, title, news, 'ET Market'])
            else:
                continue
        except:
            continue
    upperframe.extend(frame)
    
    df = pd.DataFrame(upperframe, columns=['date', 'heading', 'article_content','source'])
    return df


# In[9]:





# In[11]:


# getPastWeekNews('https://economictimes.indiatimes.com/power-grid-corporation-of-india-ltd/stocksupdate/companyid-4628.cms')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from datetime import *

import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[2]:


def latest_article(url):
    news_link=[]
    driver = webdriver.Chrome()
    driver.get(str(url))
    
    list_div_news=driver.find_elements_by_class_name("g_14bl")
    for ele in list_div_news:
        news_link.append(ele.get_attribute('href'))
    driver.quit()
    
    
    
    
    df = pd.DataFrame( columns = ['date', 'heading','article_content','source']) 


    for URL in news_link:
#         print(URL)
        text=""
        r = requests.get(URL) 
        soup = BeautifulSoup(r.content, 'html5lib')
        article_date=soup.find(class_ = "article_schedule")
        news_date=article_date.text.split("/")[0]
        news_date.lstrip("  ")
        news_date.rstrip("  ")
#         print(news_date)
        news_date= datetime.strptime(news_date,"  %B %d, %Y  ")
        today = datetime.today()
        oneweekago= today - timedelta(days = 7)
        if  news_date >= oneweekago:
            article_heading = soup.find(class_ = "article_title artTitle")
            try:
                if True: #company_name in article_heading.text:
                    article_content= soup.find(class_="content_wrapper arti-flow")
                    for a in article_content.find_all("p"):
                        text=text + a.text;    


                    data= {
                        "date" : [article_date.text],
                        "heading": [article_heading.text],
                        "article_content": [text],
                        "source":["Money Control"]
                    }

                    df1 = pd.DataFrame(data)
#                     print(df1)
                    df = pd.concat([df, df1], ignore_index = True) 
#                     print (df)
            except:
                pass
            
        else:
            break
    
    return df
    
    


# In[6]:


# b=latest_article("https://www.moneycontrol.com/company-article/infosys/news/IT")


# In[5]:


# b


# In[ ]:





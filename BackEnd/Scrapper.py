#!/usr/bin/env python
# coding: utf-8

# In[29]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from datetime import *


# In[176]:


driver = webdriver.Chrome()
main_url="https://www.moneycontrol.com/company-article/infosys/news/IT"
company_name="Infosys"
driver.get(main_url)


# In[177]:


list_div_pages_link=[main_url]
list_div_pages=driver.find_elements_by_class_name("pages")
try:
    for i in range(2,6):
        ele=driver.find_element_by_link_text(str(i))
        list_div_pages_link.append(ele.get_attribute('href'))
except:
    pass

driver.quit()


# In[178]:


list_div_pages_link


# In[179]:



tata_motors_news_link=[]

for url in list_div_pages_link:
    print(url)
    driver = webdriver.Chrome()
    driver.get(str(url))
    
    list_div_news=driver.find_elements_by_class_name("g_14bl")
    for ele in list_div_news:
        tata_motors_news_link.append(ele.get_attribute('href'))
    driver.quit()
    







    


# In[ ]:





# In[180]:


tata_motors_news_link


# In[4]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[182]:



df = pd.DataFrame( columns = ['date', 'heading','article_content']) 


for URL in tata_motors_news_link:
    print(URL)
    text=""
    r = requests.get(URL) 
    soup = BeautifulSoup(r.content, 'html5lib')
    article_date=soup.find(class_ = "article_schedule")
    article_heading = soup.find(class_ = "article_title artTitle")
    try:
        if company_name in article_heading.text:
            article_content= soup.find(class_="content_wrapper arti-flow")
            for a in article_content.find_all("p"):
                text=text + a.text;    


            data= {
                "date" : [article_date.text],
                "heading": [article_heading.text],
                "article_content": [text]
            }

            df1 = pd.DataFrame(data)
            df = pd.concat([df, df1], ignore_index = True)   
    except:
        pass
    
    


# In[ ]:





# In[184]:


df.to_csv(company_name+'.csv') 


# In[65]:


def latest_article(url):
    news_link=[]
    driver = webdriver.Chrome()
    driver.get(str(url))
    
    list_div_news=driver.find_elements_by_class_name("g_14bl")
    for ele in list_div_news:
        news_link.append(ele.get_attribute('href'))
    driver.quit()
    
    
    
    
    df = pd.DataFrame( columns = ['date', 'heading','article_content']) 


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
                        "article_content": [text]
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
    
    


# In[63]:


b=latest_article("https://www.moneycontrol.com/company-article/infosys/news/IT")


# In[64]:


b


# In[66]:


b=b.to_dict()


# In[67]:


b


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

print(torch.__version__)

# specify GPU
device = torch.device("cpu")
print (device)
print("New")

# In[18]:


df=pd.read_csv("Tata.csv")


# In[19]:


test_text=df['article_content']


# In[21]:


# test_text


# In[22]:


# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# In[24]:


# # get length of all the messages in the train set
# seq_len = [len(i.split()) for i in train_text]

# pd.Series(seq_len).hist(bins = 30)


# In[25]:


max_seq_len = 32


# In[26]:


# tokenize and encode sequences in the training set
# tokens_train = tokenizer.batch_encode_plus(
#     train_text.tolist(),
#     max_length = max_seq_len,
#     pad_to_max_length=True,
#     truncation=True,
#     return_token_type_ids=False
# )

# # tokenize and encode sequences in the validation set
# tokens_val = tokenizer.batch_encode_plus(
#     val_text.tolist(),
#     max_length = max_seq_len,
#     pad_to_max_length=True,
#     truncation=True,
#     return_token_type_ids=False
# )

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)


# In[28]:


# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
# test_y = torch.tensor(test_labels.tolist())


# In[29]:


# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False


# In[30]:


class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x


# In[31]:


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)


# In[32]:


# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 2e-5)


# In[33]:


#load weights of best model
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))


# In[34]:


# get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()


# In[35]:


# model's performance
preds = np.argmax(preds, axis = 1)
# print(classification_report(test_y, preds))


# In[36]:


# confusion matrix
# pd.crosstab(test_y, preds)
# preds


# In[60]:


def prediction(df):
    test_text=df['article_content']
    tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False)
    
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    
    
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
    
    preds = np.argmax(preds, axis = 1)
#     values, counts = np.unique(preds, return_counts=True)
#     preds_dict={}
#     for i in values: 
        
        
#         preds_dict[values[i]] = counts[i]
        
    
    
    return preds
    


# In[61]:


# preds=prediction(df)


# In[62]:





# In[54]:





# In[55]:





# In[56]:





# In[ ]:





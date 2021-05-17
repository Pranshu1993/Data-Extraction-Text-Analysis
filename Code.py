#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import nltk


# In[2]:


nltk.download("stopwords")


# In[3]:


df = pd.read_excel(r'C:\Users\ACER\Downloads\cik_list.xlsx')
df.head()
df['new_column_name'] = 'https://www.sec.gov/Archives/'
df['SECFNAME_NEW'] = df['new_column_name'] + df['SECFNAME']

del df['new_column_name']
df.head()


# In[4]:


df_positive = pd.read_excel("LoughranMcDonald_MasterDictionary_2018.xlsx")
df_positive.head()


# In[5]:


# Positive Words
df_positive_values = df_positive[df_positive['Positive']!=0]
positive_words = df_positive_values['Word'].to_list()
positive_words


# In[6]:


# Negative Words
df_negative_values = df_positive[df_positive['Negative']!=0]
negative_words = df_negative_values['Word'].to_list()


# In[ ]:


#Uncertainty List
df_uncertainty = pd.read_excel(r'C:\Users\ACER\Downloads\uncertainty_dictionary.xlsx')
uncertainty_list = df_uncertainty['Word'].tolist()

#Constraining List
df_constraining = pd.read_excel(r'C:\Users\ACER\Downloads\constraining_dictionary.xlsx')
constraining_list = df_constraining['Word'].tolist()


# In[7]:


#Stopwords
df_stopwords = pd.read_csv(r'C:\Users\ACER\Downloads\StopWords_Generic.txt', header = None)
stop_words = df_stopwords[0].tolist()


# In[8]:


# Create New Columns For Output File
df['positive_score'] = ""
df['negative_score'] = ""
df['polarity_score'] = ""
df['average_sentence_length'] = ""
df['percentage_of_complex_words'] = ""
df['fog_index'] = ""
df['complex_word_count'] = ""
df['word_count'] = ""
df['uncertainty_score'] = ""
df['constraining_score'] = ""
df['positive_word_proportion'] = ""
df['negative_word_proportion'] = ""
df['uncertainty_word_proportion'] = ""
df['constraining_word_proportion'] = ""
df['constraining_words_whole_report'] = ""


# In[9]:


def tokenize(text):
    text = re.sub(r'[^A-Za-z]',' ',text.upper())
    tokenized_words = word_tokenize(text)
    return tokenized_words


def count(dictionary, words):
    score = 0
    for i in words:
        if(i in dictionary):
            score = score+1
    return score

def remove_stopwords(words, stop_words):
    return [x for x in words if x not in stop_words]

def complex_words(text):
    vowels = ['A','E','I','O','U']
    cw = 0
    for x in text:
        count = 0
        for i in range(0,len(x)-1):
            if x[i] in vowels and x[i+1] not in vowels:
                count = count + 1
                if x.endswith("E") or x.endswith("ED"):
                    count -= 1
        if count > 2:
            cw = cw + 1
    return cw


# In[10]:


i= 0
for URL in df['SECFNAME_NEW']:
    
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'html5lib')

    #   print(soup.prettify())
    #   doc_length = len(soup.get_text())

    txt = str(soup.get_text())
    text = txt.lower()

    tokenized_text = tokenize(text)
    # print(tokenized_text)

    removed_stopw_tokenized = remove_stopwords(tokenized_text,stop_words)
    num_words = len(removed_stopw_tokenized) 

    # len(removed_stopw_tokenized)
    
    # Positive Score
    positive_score = count(positive_words,removed_stopw_tokenized)
    #     print(positive_score)
    df.at[i,'positive_score'] = positive_score
    
    # Negative Score
    negative_score = count(negative_words,removed_stopw_tokenized)
    #     print(negative_score)
    df.at[i,'negative_score'] = negative_score
    
    #Polarity [-1 to +1]
    polarity_score = (positive_score - negative_score)/ ((positive_score + negative_score) + 0.000001)
    #     print(polarity_score)
    df.at[i,'polarity_score'] = polarity_score
    
    #Average Sentence length
    sent_tok = sent_tokenize(text)
    sent_tok
    avg_sent_len = num_words/len(sent_tok)
    #     print(avg_sent_len)
    df.at[i,'average_sentence_length'] = avg_sent_len

    # % of complex words
    total_complex_words = complex_words(removed_stopw_tokenized)
    #     print(total_complex_words)
    complex_word_perc = total_complex_words/len(removed_stopw_tokenized)*100
    df.at[i,'percentage_of_complex_words'] = complex_word_perc

    df.at[i,'complex_word_count'] = total_complex_words
    
    # Word count 
    #   print(num_words)
    df.at[i,'word_count'] = num_words
    # Fog index
    fog_index = (avg_sent_len + complex_word_perc)*0.4
    #     fog_index
    df.at[i,'fog_index'] = fog_index

    # Uncertainty Score
    
    uncertainty_score = 0

    for x in removed_stopw_tokenized:
        if(x in uncertainty_list):
            uncertainty_score += 1

    #   print(uncertainty_score)
    df.at[i,'uncertainty_score'] = uncertainty_score

    #Constraining Score
    constraining_score = 0

    for x in removed_stopw_tokenized:                   
        if(x in constraining_list):
            constraining_score = constraining_score+1

    df.at[i,'constraining_score'] = constraining_score

    # Positive proportion
    positive_word_proportion = round((positive_score/num_words),2)
    df.at[i,'positive_word_proportion'] = positive_word_proportion

    # Negative proportion
    negative_word_proportion = round((negative_score/num_words),2)
    df.at[i,'negative_word_proportion'] = negative_word_proportion

    # Uncertainty Proportion
    uncertainty_word_proportion = round((uncertainty_score/num_words),2)
    #     print(uncertainty_word_proportion)
    df.at[i,'uncertainty_word_proportion'] = uncertainty_word_proportion

        # Constraining Proportion
    constraining_word_proportion = round((constraining_score/num_words),2)
    #     print(constraining_word_proportion)
    df.at[i,'constraining_word_proportion'] = constraining_word_proportion

    # constraining words whole report
    constraining_words_whole_report = 0

    for x in removed_stopw_tokenized:
        if x in constraining_list:
            constraining_words_whole_report = 1+ constraining_words_whole_report
    #   print(constraining_words_whole_report)
    df.at[i,'constraining_words_whole_report'] = constraining_words_whole_report
    
    i = i + 1


# In[15]:


df.head()
df.to_csv(r'C:\Users\ACER\OneDrive\Documents\Anuj\output.csv')

#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


# In[20]:


df = pd.read_csv('C:\\Users\\JEBINA P\\Downloads\\dataset_covid.csv')
df.head()


# In[4]:


df.dtypes


# In[22]:


df[df.duplicated()]


# In[23]:


df.head(7)


# In[24]:


df.info()


# In[5]:


{col : df[col].unique() for col in df if df[col].dtype == object}


# In[6]:


df = df.astype(
    {
        'Country/Region' : 'category',
        'WHO Region' : 'category',
    }
)


# In[7]:


df.dtypes


# In[25]:


df['WHO Region'].unique()


# In[8]:


df.corr()


# In[32]:


dataframe=df[['WHO Region','Confirmed','Deaths', 'Recovered', 'Active']].groupby('WHO Region').mean().round(2).sort_values(by='Confirmed', ascending=False)


# In[33]:


dataframe


# In[34]:


perc_list =[]
list1=[]
list2=[]
for i in range(len(dataframe)):
    perc_recovered=round(dataframe['Recovered'][i]*100/dataframe['Confirmed'][i], 2)
    perc_death= round((dataframe['Deaths'][i]*100)/dataframe['Confirmed'][i],2)
    perc_active = round((dataframe['Active'][i]*100)/dataframe['Confirmed'][i],2)
    perc_list.append(perc_recovered)
    list1.append(perc_death)
    list2.append(perc_active)
dataframe['perc_recovered'] =perc_list
dataframe['perc_active']=list2
dataframe['perc_death']= list1


# In[35]:


dataframe=dataframe.reset_index()
dataframe


# In[36]:


data=dataframe[['WHO Region', 'Confirmed']]
data


# In[37]:


plt.pie(data['Confirmed'], labels=data['WHO Region'], shadow=True, autopct="%0.2f%%", radius=1.2, labeldistance=1.05)


# In[38]:


frame=df[['Country/Region','Confirmed', 'Deaths', 'Recovered', 'Active']].groupby('Country/Region').mean().round(2).sort_values(by='Confirmed', ascending=False).reset_index()
frame


# In[40]:


data=frame[0:11]
data


# In[41]:


plt.pie(data['Confirmed'], labels=data['Country/Region'], autopct="%0.2f%%",pctdistance=0.8)


# In[43]:


frame1=df[['Country/Region', 'Recovered',]].groupby('Country/Region').mean().round(2).sort_values(by='Recovered', ascending=False).reset_index()[0:11]
frame1


# In[44]:


plt.pie(frame1['Recovered'], labels=frame1['Country/Region'], autopct="%0.2f%%", pctdistance=0.8)


# In[45]:


frame1=df[['Country/Region', 'Deaths',]].groupby('Country/Region').mean().round(2).sort_values(by='Deaths', ascending=False).reset_index()[0:10]
plt.pie(frame1['Deaths'], labels=frame1['Country/Region'], autopct="%0.2f%%", pctdistance=0.8)


# In[46]:


frame1=df[['Country/Region', 'Confirmed',]].groupby('Country/Region').mean().round(2).sort_values(by='Confirmed', ascending=True).reset_index()[0:10]
plt.pie(frame1['Confirmed'], labels=frame1['Country/Region'], autopct="%0.2f%%", pctdistance=0.8)


# In[48]:


frame= df[['Country/Region', 'Deaths / 100 Cases']].sort_values(by='Deaths / 100 Cases', ascending=False)[0:10]
sns.barplot(y='Country/Region', x='Deaths / 100 Cases', data=frame)


# In[49]:


frame= df[['Country/Region', 'Recovered / 100 Cases']].sort_values(by='Recovered / 100 Cases', ascending=False)[0:10]
sns.barplot(y='Country/Region', x='Recovered / 100 Cases', data=frame)


# In[50]:


frame=df[['Country/Region', 'New cases']].sort_values(by='New cases', ascending=False)[0:10]
sns.barplot(x='New cases', y='Country/Region', data=frame)


# In[51]:


frame=df[['Country/Region', 'New deaths']].sort_values(by='New deaths', ascending=False)[0:10]
sns.barplot(x='New deaths', y='Country/Region', data=frame)


# In[9]:


df.corr().style.background_gradient(cmap='winter')


# In[10]:


correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[11]:


abs(df.corr())[['New cases']].style.background_gradient(cmap='Reds')


# In[12]:


df.corr().style.highlight_max(axis=0)


# In[13]:


df.corr().style.highlight_min(axis=0)


# In[14]:


df.describe(include=['category'])


# In[15]:


df.describe()


# In[16]:


df.isna().sum()


# In[17]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recovered', y='Deaths', hue='WHO Region', data=df)
plt.title("Recovered vs. Deaths")
plt.xlabel("Recovered Cases")


plt.ylabel("Deaths")
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
sns.countplot(x='WHO Region', data=df)
plt.title("Distribution of WHO Regions")
plt.xticks(rotation=45)
plt.show()


# In[52]:


df.head(3)


# In[53]:


df[['WHO Region','Country/Region', 'Confirmed last week']].groupby(['WHO Region','Country/Region']).mean().round(2).sort_values(by='Confirmed last week', ascending=False)[0:10]


# In[54]:


Frame= df[['WHO Region','Country/Region', 'Confirmed last week']].groupby(['WHO Region','Country/Region']).mean().round(2).sort_values(by='Confirmed last week', ascending=False).reset_index()[0:10]


# In[55]:


sns.barplot(x='Confirmed last week', y='Country/Region', data=Frame)


# In[ ]:





# In[ ]:





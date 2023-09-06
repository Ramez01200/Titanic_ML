#!/usr/bin/env python
# coding: utf-8

# ### Titanic: Machine Learning

# In[3]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC


# ### Data Dictionary
# * Survived: 0 = No, 1 = Yes
# * pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# * sibsp: # of siblings / spouses aboard the Titanic
# * parch: # of parents / children aboard the Titanic
# * ticket: Ticket number
# * cabin: Cabin number
# * embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# In[4]:


train = pd.read_csv("C:/Users/Ramez/Downloads/New folder/train.csv")
test = pd.read_csv("C:/Users/Ramez/Downloads/New folder/test.csv")
train_test_data = [train, test]


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.shape


# In[8]:


test.shape


# In[9]:


train.info()


# In[10]:


test.info()


# In[11]:


train.isnull().sum()


# In[12]:


test.isnull().sum()


# In[13]:


def barchart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead =  train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[14]:


barchart('Sex')


# The Chart confirms Women more likely survivied than Men

# In[15]:


barchart('Pclass')


# The Chart confirms 1st class more likely survivied than other classes
# 
# The Chart confirms 3rd class more likely dead than other classes

# In[16]:


barchart('SibSp')


# The Chart confirms a person aboarded with more than 2 siblings or spouse more likely survived
# 
# The Chart confirms a person aboarded without siblings or spouse more likely dead

# In[17]:


barchart('Parch')


# The Chart confirms a person aboarded with more than 2 parents or children more likely survived
# 
# The Chart confirms a person aboarded alone more likely dead

# In[18]:


barchart('Embarked')


# The Chart confirms a person aboarded from C slightly more likely survived
# 
# The Chart confirms a person aboarded from Q more likely dead
# 
# The Chart confirms a person aboarded from S more likely dead

# In[19]:


train.head()


# In[20]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    


# In[21]:


train['Title'].value_counts()


# In[22]:


test['Title'].value_counts()


# ## Title map
# Mr : 0 
# 
# Miss : 1
# 
# Mrs: 2
# 
# Others: 3

# In[24]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[25]:


train.head()


# In[26]:


test.head()


# In[27]:


barchart('Title')


# In[28]:


train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)


# In[29]:


train.head()


# In[30]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1})


# In[31]:


train.head()


# In[32]:


barchart('Sex')


# ## Age
# 
# some age is missing
# 
# Let's use Title's median age for missing Age

# In[34]:


train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'),inplace=True)


# In[35]:


train['Age'].info()


# In[36]:


train['Age'].value_counts()


# In[37]:


train.info()


# In[38]:


test.info()


# feature vector map:
# 
# child: 0
# 
# young: 1
# 
# adult: 2
# 
# mid-age: 3
# 
# senior: 4

# In[40]:


for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16 , 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train.head()


# In[41]:


train['Age'].value_counts()


# In[42]:


barchart('Age')


# In[43]:


train['Embarked'].value_counts()


# In[44]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[45]:


train.info()


# In[46]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2})


# In[48]:


train['Embarked'].value_counts()


# In[49]:


train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)


# In[50]:


train.info()


# In[51]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[52]:


train['Fare'].value_counts()


# In[53]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[54]:


train.head()


# In[55]:


features_drop = ['Ticket', 'SibSp', 'Parch','Cabin']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)


# In[56]:


train = train.drop(['PassengerId'], axis=1)


# In[57]:


train.head()


# In[58]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[60]:


# Logistic Regression


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[61]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[63]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[64]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 
              'Random Forest', 'Decision Tree'],
    'Score': [ acc_log, 
              acc_random_forest,  acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[65]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })


# In[66]:


submission.head()


# In[67]:


submission.shape


# In[68]:


submission.to_excel("submission.xlsx", index=False)


# In[ ]:





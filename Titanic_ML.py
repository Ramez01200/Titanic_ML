#!/usr/bin/env python
# coding: utf-8

# In[69]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[3]:


train = pd.read_csv("C:/Users/Ramez/Downloads/New folder/train.csv")
test = pd.read_csv("C:/Users/Ramez/Downloads/New folder/test.csv")
train_test_data = [train, test]


# In[4]:


train.head()


# In[5]:


test.head()


# In[7]:


train.shape


# In[8]:


test.shape


# In[9]:


train.info()


# In[10]:


test.info()


# In[12]:


train.isnull().sum()


# In[13]:


test.isnull().sum()


# In[14]:


def barchart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead =  train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[15]:


barchart('Sex')


# In[17]:


barchart('Pclass')


# In[18]:


barchart('SibSp')


# In[19]:


barchart('Parch')


# In[20]:


barchart('Embarked')


# In[21]:


train.head()


# In[22]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    


# In[23]:


train['Title'].value_counts()


# In[24]:


test['Title'].value_counts()


# In[25]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[26]:


train.head()


# In[27]:


test.head()


# In[28]:


barchart('Title')


# In[31]:


train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)


# In[32]:


train.head()


# In[34]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1})


# In[35]:


train.head()


# In[36]:


barchart('Sex')


# In[37]:


train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'),inplace=True)


# In[39]:


train.head(30)
train.groupby("Title")["Age"].transform("median")


# In[40]:


train.info()


# In[41]:


test.info()


# In[44]:


for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16 , 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train.head()


# In[45]:


barchart('Age')


# In[46]:


train['Embarked'].value_counts()


# In[47]:


for dataset in train_test_data:
    dataset['Embarked']= dataset['Embarked'].fillna('S')


# In[48]:


train.info()


# In[49]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2})


# In[51]:


train.head()


# In[52]:


train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)


# In[53]:


train.head()


# In[54]:


train.info()


# In[56]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[57]:


train.head()


# In[58]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[59]:


train.head()


# In[75]:


features_drop = ['Ticket', 'SibSp', 'Parch','Cabin']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)


# In[76]:


train = train.drop(['PassengerId'], axis=1)


# In[77]:


train.head()


# In[78]:


X_train0 = train.drop("Survived", axis=1)
Y_train0 = train["Survived"]
X_test0  = test.drop("PassengerId", axis=1).copy()
X_train0.shape, Y_train0.shape, X_test0.shape


# In[81]:


logreg = LogisticRegression()
logreg.fit(X_train0, Y_train0)
Y_pred = logreg.predict(X_test0)
acc_log = round(logreg.score(X_train0, Y_train0) * 100, 2)
acc_log


# In[82]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train0, Y_train0)
Y_pred = decision_tree.predict(X_test0)
acc_decision_tree = round(decision_tree.score(X_train0, Y_train0) * 100, 2)
acc_decision_tree


# In[83]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train0, Y_train0)
Y_pred = random_forest.predict(X_test0)
random_forest.score(X_train0, Y_train0)
acc_random_forest = round(random_forest.score(X_train0, Y_train0) * 100, 2)
acc_random_forest


# In[84]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 
              'Random Forest', 'Decision Tree'],
    'Score': [ acc_log, 
              acc_random_forest,  acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[93]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })


# In[94]:


submission.head()


# In[95]:


submission.shape


# In[96]:


display(submission)


# In[97]:


submission.to_excel("submission.xlsx", index=False)


# In[ ]:





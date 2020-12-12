#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ----------Jiaqi Chen-------------------
#----------Fianl project------------------
#----------SVM--------------
from sklearn import svm
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.model_selection import GridSearchCV
 
def readimage(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img
 
def readlabel(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab
 

    #train_data  = readimage("train-images.idx3-ubyte")
    #train_label = readlabel("train-labels.idx1-ubyte")
 


# In[2]:


train_data  = readimage("train-images.idx3-ubyte")
train_label = readlabel("train-labels.idx1-ubyte")
train_data  = train_data[:10000,:]
#print(np.shape(train_data))
train_label = train_label[:10000]
test_data   = readimage("t10k-images.idx3-ubyte")
test_label  = readlabel("t10k-labels.idx1-ubyte")
#print(np.shape(test_data))
test_data  = test_data[:1000,:]
#print(np.shape(train_data))
test_label = test_label[:1000]
svc=svm.SVC()
parameters = {'kernel':['rbf'], 'C':[1]}
#parameters = {'kernel':['sigmoid'], 'degree':[5],'C':[1]}
print("Train...")
clf=GridSearchCV(svc,parameters,n_jobs=-1)
start = time()
clf.fit(train_data, train_label)
end = time()
t = end - start
print('Train：%dmin%.3fsec' % (t//60, t - 60 * (t//60)))
prediction = clf.predict(test_data)
print("accuracy: ", accuracy_score(prediction, test_label))
accurate=[0]*10
sumall=[0]*10
#rint(np.shape(accurate))
i=0
while i<len(test_label):
    sumall[test_label[i]]+=1
    if prediction[i]==test_label[i]:
        accurate[test_label[i]]+=1
    i+=1
    
print("Correct number of each label：",accurate)
print("Total test number of each label：",sumall)
 


# In[3]:


#len(test_label)


# In[4]:


#test_label


# In[5]:


#np.shape(test_label)


# In[6]:


#accurate


# In[ ]:





# In[ ]:





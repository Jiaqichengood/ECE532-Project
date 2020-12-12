#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ----------Jiaqi Chen-------------------
#----------Fianl project------------------
#----------Ridge regression--------------
import numpy as np
from time import time
from struct import unpack

def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img

def __read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab
    
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

def load_mnist(train_image_path, train_label_path, test_image_path, test_label_path, normalize=True, one_hot=True):
    image = {
        'train' : __read_image(train_image_path),
        'test'  : __read_image(test_image_path)
    }

    label = {
        'train' : __read_label(train_label_path),
        'test'  : __read_label(test_label_path)
    }
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

   # return (image['train'], label['train']), (image['test'], label['test'])
    return image['train'], label['train'], image['test'], label['test']


# In[2]:


train_size = 10000
test_size = 1000


# In[3]:


traindata_all,trainlabel_all,testdata,testlabel=load_mnist('trainimages.idx3-ubyte','trainlabels.idx1-ubyte','t10kimages.idx3-ubyte','t10klabels.idx1-ubyte')


# In[4]:


#traindata_all,trainlabel_all,testdata,testlabel=load_mnist('D:/Final project/trainimages.idx3-ubyte','D:/Final project/trainlabels.idx1-ubyte','D:/Final project/t10kimages.idx3-ubyte','D:/Final project/t10klabels.idx1-ubyte')


# In[5]:


traindata=traindata_all[:train_size,:]


# In[6]:


trainlabel=trainlabel_all[:train_size,:]


# In[7]:


col_num = traindata.shape[1]
row_num = traindata.shape[0]
#w_opt = np.linalg.inv(traindata.transpose()@traindata)@traindata.transpose()@trainlabel
lamb = 1e0
#lamb = 1e2


# In[8]:


# 原公式
I = np.eye(col_num)
lam_I = lamb*I
#w_opt = traindata.transpose()@np.linalg.inv(traindata@traindata.transpose()+lam_I)@trainlabel
print("Train...")
start = time()

w_opt = np.linalg.inv(traindata.transpose()@traindata+lam_I)@traindata.transpose()@trainlabel

end = time()
t = end - start
print('Train：%dmin%.3fsec' % (t//60, t - 60 * (t//60)))
#y_hat = np.sign(x_eval@w_opt)


# In[9]:


testdata=testdata[:test_size,:]


# In[10]:


testlabel = testlabel[:test_size,:]


# In[11]:


#test_hat = np.sign(testdata@w_opt )
test_hat = testdata@w_opt 


# In[12]:


#test中正确的数
testlabel_index = np.argmax(testlabel, axis=1)


# In[13]:


#test中预测的数
test_hat_index = np.argmax(test_hat, axis=1)


# In[14]:


test_hat_index = np.array([test_hat_index])


# In[15]:


testlabel_index = np.array([testlabel_index])


# In[16]:


testlabel_index = testlabel_index.transpose()


# In[17]:


test_hat_index = test_hat_index.transpose()


# In[18]:


error_vec = [0 if i[0]==i[1] else 1 for i in np.hstack((testlabel_index, test_hat_index))]


print('Errors number: '+ str(sum(error_vec)))
print('Errors rate: ',(sum(error_vec))/test_size)

accuracy = (1000-sum(error_vec))/10
print('accuracy: ',accuracy)


# In[ ]:





# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Mon May 10 04:24:07 2021

@author: Saif
"""

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
from keras.preprocessing import image                  
from keras.preprocessing.image import img_to_array, load_img
from PIL import ImageFile                           
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from train import train

model, test_tensors, test_targets = train()

model.load_weights('saved_models/weights.best.with_augmentation_new.hdf5')
#model.load_weights('saved_models/weights.best.with_augmentation.hdf5')
# get index of predicted alphabetnfor each image in test set
alphabet_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

from sklearn import metrics
y_true = [np.argmax(y_test) for y_test in test_targets]
f1_accuracy = 100* metrics.f1_score(y_true,alphabet_predictions, average = 'micro')
print('Test F1 accuracy: %.4f%%' % f1_accuracy)


# In[63]:


def trial_prediction(img_path):
    img = load_img(img_path, target_size=(64, 64), grayscale=True)
    x = img_to_array(img)
    tensor = x #np.expand_dims(x, axis=0)
    test_img = np.expand_dims(tensor, axis=0)
    prediction_idx = np.argmax(model.predict(test_img))
    alphbt = ['অ','আ','ই','ঈ','উ','ঊ',
              'ঋ','এ','ঐ','ও','ঔ',
              'ক','খ','গ','ঘ','ঙ',
              'চ','ছ','জ','ঝ','ঞ',
              'ট','ঠ','ড','ঢ','ণ',
              'ত','থ','দ','ধ','ন',
              'প','ফ','ব','ভ','ম',
              'য','র','ল',
              'শ','ষ','স','হ',
              'ড়','ঢ়','য়',
              'ৎ','ং','ঃ','ঁ'
              'ঁ']
    output = alphbt[prediction_idx]
    return output

#%% run this only
out = trial_prediction('trial/ka.jpg')
print(out)


# ### Visualizations and Error Analysis
# This may give you some insight into why the network is misclassifying certain objects.

# In[64]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, alphabet_predictions)

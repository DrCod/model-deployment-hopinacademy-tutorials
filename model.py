import scipy.io 
import numpy as np
from sklearn.utils import shuffle 
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import os
from PIL import Image
from scipy import misc
import re
from skimage import color
from skimage import io



X_test =[]
y_test =[]


ims = [file for file in os.listdir('test-imgs/')]

for img in ims[1:]:
    if '11' in img: y_test.append(1)
    if '2' in img: y_test.append(2)
    if '3' in img: y_test.append(3)
    if '4' in img: y_test.append(4)
    if '5' in img: y_test.append(5)
    if '6' in img: y_test.append(6)
    if '7' in img: y_test.append(7)
    if '8' in img: y_test.append(8)
    if '9' in img: y_test.append(9)
    if '10' in img: y_test.append(10)
    
    img = misc.imread('test-imgs/'+img)
    img = color.rgb2gray(img)
    img_arr = np.asarray(img)
    X_test.append(img_arr)
    
        
X =np.asarray(X_test)
X = X / 255.
X = X.reshape(X.shape[0],X.shape[1]* X.shape[2])
y =np.asarray(y_test)

    
mlp = MLPClassifier(hidden_layer_sizes=(50), max_iter=50, alpha=1e-2,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X, y) 

# save model
joblib.dump(mlp, 'model.pkl')

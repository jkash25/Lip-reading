
import pandas as pd
import os
import imutils
import time
from keras.utils import np_utils, generic_utils
import shutil
from skimage.transform import resize
from sklearn.utils import shuffle
import tensorflow
import keras
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Activation, ZeroPadding3D, TimeDistributed, LSTM, GRU, Reshape
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
import numpy as np
import os
import imutils
import dlib 
import cv2 
import imageio
from imutils import face_utils
from google.colab import drive


drive.mount('/content/gdrive')
!unzip '/content/gdrive/MyDrive/Colab Notebooks/miracl.zip' > /dev/null

def crop_and_save_image(img, img_path, write_img_path, img_name):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('/content/shape_predictor_68_face_landmarks.dat')
  image = cv2.imread(img_path)
  image = imutils.resize(image,width=500)
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  rects = detector(gray,1)
  if len(rects)>1:
    print("ERROR: more than one face detected")
    return
  if len(rects)<1:
    print("ERROR: no faces detected")
    return
  for (i,rect) in enumerate(rects):
    shape = predictor(gray,rect)
    shape = face_utils.shape_to_np(shape)
    name,i,j = 'mouth',48,68
    (x,y,w,h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = gray[y:y+h, x:x+w]
    roi = imutils.resize(roi,width=250,inter=cv2.INTER_CUBIC)
    print(rects[i])
    
    people = ['F01','F02','F04','F05','F06','F07','F08','F09', 'F10','F11','M01','M02','M04','M07','M08']
data_types = ['words']
folder_enum = ['01','02','03','04','05','06','07','08', '09', '10']
instances = ['01','02','03','04','05','06','07','08', '09', '10']

words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']          
words_di = {i:words[i] for i in range(len(words))}

def crop_one_person():      
    #os.mkdir('cropped')
    people = ['F01']
    data_types = ['words']
    folder_enum = ['01']
    instances = ['01']

    i = 1
    for person_ID in people:
        if not os.path.exists('/content/cropped/cropped/' + person_ID ):
            os.mkdir('/content/cropped/cropped/' + person_ID + '/')

        for data_type in data_types:
            if not os.path.exists('/content/cropped/cropped/' + person_ID + '/' + data_type):
                os.mkdir('/content/cropped/cropped/' + person_ID + '/' + data_type)

            for phrase_ID in folder_enum:
                if not os.path.exists('/content/cropped/cropped/' + person_ID + '/' + data_type + '/' + phrase_ID):
                    # F01/phrases/01
                    os.mkdir('/content/cropped/cropped/' + person_ID + '/' + data_type + '/' + phrase_ID)

                for instance_ID in instances:
                    # F01/phrases/01/01
                    directory = '/content/dataset/dataset' +'/'+ person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'
                    dir_temp = person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'
                    print(directory)
                    filelist = os.listdir(directory)
                    if not os.path.exists('/content/cropped/cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID):
                        os.mkdir('/content/cropped/cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID)

                        for img_name in filelist:
                            if img_name.startswith('color'):
                                image = imageio.imread(directory + '' + img_name)
                                crop_and_save_image(image, directory + '' + img_name,
                                                    dir_temp + '' + img_name, img_name)

    print(f'Iteration : {i}')
    i += 1
    #shutil.rmtree('/content/drive/MyDrive/Colab Notebooks/cropped/cropped')
    
    times = 0
for _ in range(7):
     t1 = time.time()
     crop_one_person()
     t2 = time.time()
     times += (t2 - t1)

print("Average time over 7 iterations : ", times/7)

max_seq_length = 22

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []


MAX_WIDTH = 100
MAX_HEIGHT = 100


t1 = time.time()
UNSEEN_VALIDATION_SPLIT = ['F07', 'M02']
UNSEEN_TEST_SPLIT = ['F04']



directory = "/content/dataset/dataset"
#directory = '/content/cropped/cropped'
for person_id in people:
    tx1 = time.time()
    for data_type in data_types:
        for word_index, word in enumerate(folder_enum):
            #print(f"Word : '{words[word_index]}'")
            for iteration in instances:
                path = os.path.join(directory, person_id, data_type, word, iteration)
                #print(path)
                filelist = sorted(os.listdir(path + '/'))
                #print(filelist)
                sequence = [] 
                for img_name in filelist:
                    if img_name.startswith('color'):
                        #image = cv2.imread(path+'/'+img_name)
                        #cv2_imshow(image)
                        image = imageio.imread(path + '/' + img_name)
                        image = np.resize(image,(MAX_WIDTH,MAX_HEIGHT))
                        
                        #image = resize(image, (MAX_WIDTH, MAX_HEIGHT))
                        image = 255 * image
                        # Convert to integer data type pixels.
                        image = image.astype(np.uint8)
                        sequence.append(image)                        
                pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]
                                            
                sequence.extend(pad_array * (max_seq_length - len(sequence)))
                sequence = np.array(sequence)
                                
                if person_id in UNSEEN_TEST_SPLIT:
                    X_test.append(sequence)
                    y_test.append(word_index)
                elif person_id in UNSEEN_VALIDATION_SPLIT:
                    X_val.append(sequence)
                    y_val.append(word_index)
                else:
                    #print('shape: ',sequence.shape)
                    X_train.append(sequence)
                    y_train.append(word_index)    
    tx2 = time.time()
    print(f'Finished reading images for person {person_id}. Time taken : {tx2 - tx1} secs.')    
    
t2 = time.time()
print(f"Time taken for creating constant size 3D Tensors from those cropped lip regions : {t2 - t1} secs. Or {(t2-t1)/60} minutes.")


X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

def normalize_it(X):
    v_min = X.min(axis=(2, 3), keepdims=True)
    v_max = X.max(axis=(2, 3), keepdims=True)
    #print(v_min)
    #print('Max: ',v_max)
    X = (X - v_min)/(v_max - v_min)
    X = np.nan_to_num(X)
    return X
  X_train = normalize_it(X_train)
#print(X_train)
X_val = normalize_it(X_val)
X_test = normalize_it(X_test)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
y_val = np_utils.to_categorical(y_val, 10)


X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=0)
X_val, y_val = shuffle(X_val, y_val, random_state=0)

X_train = np.expand_dims(X_train, axis=4)
X_val = np.expand_dims(X_val, axis=4)
X_test = np.expand_dims(X_test, axis=4)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
  

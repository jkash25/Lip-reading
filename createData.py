import LR
import click
import captureSplit
import glob
import numpy as np
import warnings
from pathlib import Path
import pandas as pd
import os
import imutils
import dlib

# from dlib import frontal_face_detector
import cv2
import imageio
from PIL import Image
from imutils import face_utils
import time
from keras.utils import np_utils, generic_utils
import shutil
from skimage.transform import resize
from sklearn.utils import shuffle
from skimage.io import imread, imsave, imshow
import tensorflow
import keras
from keras import layers
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.layers import Activation, ZeroPadding3D, TimeDistributed, LSTM, GRU, Reshape
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

words = ['Begin','Choose','Connection','Navigation','Next', 'Previous', 'Start','Stop','Hello','Web']
words_di = {i:words[i] for i in range(len(words))}
people = ['Jai01','Jai02','Jai03','Jai04','Jai05','Jai06','Jai07','Jai08','Jai09','Jai10']
data_types = ['words']
folder_nums = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
instances = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
starting = "C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\self_training\\"
tempStarting = "C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\tempframes\\"
winname = "Recording Started"
words = [
    "Begin",
    "Choose",
    "Connection",
    "Navigation",
    "Next",
    "Previous",
    "Start",
    "Stop",
    "Hello",
    "Web",
]

words_di = {i: words[i] for i in range(len(words))}


def capture_split_for_self_training(word, iteration):
    blank_image2 = 255 * np.ones(shape=[200, 600, 3], dtype=np.uint8)
    fpsLimit = 0.1
    capture = cv2.VideoCapture(0)
    codec = cv2.VideoWriter_fourcc(*"XVID")

    recording_flag = False
    hog_face_detector = dlib.get_frontal_face_detector()

    dlib_facelandmark = dlib.shape_predictor(
        "C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\shape_predictor_68_face_landmarks.dat"
    )
    startTime = time.time()

    while True:
        ret, frame = capture.read()
        nowTime = time.time()
        if (nowTime - startTime) > fpsLimit:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = hog_face_detector(gray)
            for face in faces:

                face_landmarks = dlib_facelandmark(gray, face)

                for n in range(0, 47):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
            cv2.imshow("FRAME", frame)
            key = cv2.waitKey(1)
            if key % 256 == 27:
                break
            elif key % 256 == 32:
                
                image = cv2.putText(
                    blank_image2,
                    "Say the word: " + word + " Iteration: " + str(iteration) + "/10",
                    (0, 100),
                    fontScale=0.5,
                    color=(0, 0, 255),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                )
                cv2.namedWindow(winname)
                cv2.moveWindow(winname, 800, 40)
                cv2.imshow(winname, image)
                # cv2.imshow("Recording Started!", image)

                # print("Space here")
                if recording_flag == False:
                    output = cv2.VideoWriter("video.avi", codec, 30, (640, 480))
                    recording_flag = True
                else:
                    recording_flag = False

            if recording_flag:
                output.write(frame)
            startTime = time.time()

    capture.release()
    output.release()
    cv2.destroyAllWindows()

    newpath = "C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\tempframes"
    if os.path.exists(newpath):
        print()
        print("Directory Exists: Clearing it...")

        shutil.rmtree(newpath)

    if not os.path.exists(newpath):
        print()
        print("Directory Doesn't Exist: Creating it...")
        os.makedirs(newpath)

    video = cv2.VideoCapture("C:\\Users\\Jai K\\CS Stuff\\video.avi")
    frameNr = 0
    failed = 0
    while True:
        success, frame = video.read()
        if success and frameNr<10:
            cv2.imwrite(
                f"C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\tempframes\\frame00{frameNr}.jpg", frame)
        elif success and frameNr>=10:
            cv2.imwrite(f"C\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\tempframes\\frame0{frameNr}.jpg",frame)
        else:
            break
        frameNr += 1

    video.release()

def general_crop_for_self_training(path):
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\shape_predictor_68_face_landmarks.dat")
    frame = cv2.imread(path)
    x_arr = []
    y_arr = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copy = gray.copy()
    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(48,68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            x_arr.append(x)
            y_arr.append(y)
            xmax = max(x_arr)
            xmin = min(x_arr)
            ymax = max(y_arr)
            ymin = min(y_arr)
            cv2.circle(gray, (x, y), 1, (0, 255, 255), 1)

    copy = copy[ymin-10:ymax+10, xmin-5:xmax+5]
    scale_percent = 200
    width2 = 100
    height2 = 100
    dim = (width2, height2)
    resized_cropped = cv2.resize(copy, dim, interpolation = cv2.INTER_AREA)
    return resized_cropped

def make_dirs():
    for person in people:
        for data_type in data_types:
            for num in folder_nums:
                for instance in instances:
                    os.mkdir(starting +person+ "\\"+data_type+ "\\"+num + "\\" + instance)


def make(person,word_to_create): # pass in begin, navigation, or whichever word.

    folder_number  = words.index(word_to_create)
    word = words[folder_number] 
    if folder_number==9:
        folder_number =10
    else:
        folder_number = '0'+str(folder_number+1)

    
        for instance_index, instance in enumerate(instances):
            shutil.rmtree("C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\self_training\\"+person+"\\words"+"\\"+folder_number+"\\"+instance)
            os.mkdir("C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\self_training\\"+person+"\\words"+"\\"+folder_number+"\\"+instance)
            #print('Instance index: ',instance_index)
            print("Instance: "+instance)
            #iteration = instance_index
            capture_split_for_self_training(word, instance)
            for frame in os.listdir("C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\tempframes"):
                #print(tempStarting+frame)
                img = general_crop_for_self_training(tempStarting+frame)
                cv2.imwrite("C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\self_training\\"+person+"\\words\\"+folder_number+'\\'+instance+'\\'+frame,img)



make("Jai01","Begin")
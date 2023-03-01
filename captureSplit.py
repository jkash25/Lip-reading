from LR import *
import glob
from tkinter import *
import tkinter.messagebox
import numpy as np
import warnings
from pathlib import Path
import pandas as pd
import os
import imutils
import dlib
#from dlib import frontal_face_detector
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

# from tensorflow.keras.applications.resnet50 import ResNet50
warnings.filterwarnings("ignore")



winname = 'Recording Started'
blank_image2 = 255 * np.ones(shape=[50, 330, 3], dtype=np.uint8)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\shape_predictor_68_face_landmarks.dat")

def capture_split():
    fpsLimit = 0.1

    directions = "C:\\Users\\Jai K\\CS Stuff\\directions.jpg"
    pic = cv2.imread(directions)
    cv2.imshow("Directions", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
                image =cv2.putText(blank_image2, "Recording Started", (10, 30),fontScale=1,color=(0,0,255), fontFace=cv2.FONT_HERSHEY_TRIPLEX)
                cv2.namedWindow(winname)
                cv2.moveWindow(winname,800,40)
                cv2.imshow(winname,image)
                #cv2.imshow("Recording Started!", image)

                #print("Space here")
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

    newpath = "C:\\Users\\Jai K\\CS Stuff\\frames"
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
    failed= 0
    while True:
        success, frame = video.read()
        if success:
            cv2.imwrite(
                f"C:\\Users\\Jai K\\CS Stuff\\frames\\frame{frameNr}.jpg", frame
            )
        else:
            print("Failed to write frame")
            break
        frameNr += 1

    video.release()
    
        
def general_crop(path_to_image):
    #print("GENERAL CROP REACHED")
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\shape_predictor_68_face_landmarks.dat")
    frame = cv2.imread(path_to_image)
    x_arr = []
    y_arr = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray,width=500)
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


def crop_user():
    newpath = "C:\\Users\\Jai K\\CS Stuff\\cropped_frames"
    if os.path.exists(newpath):
        #print("Directory Exists: Clearing it...")
        shutil.rmtree(newpath)

    if not os.path.exists(newpath):
        #print("Directory Doesn't Exist: Creating it...")
        os.makedirs(newpath)

    path = "C:\\Users\\Jai K\\CS Stuff\\frames\\"
    frames = os.listdir(path)

    frameNR = 0
    for index, frame in enumerate(frames):
        try:
            image = general_crop(path + "frame" + str(index) + ".jpg")
        except UnboundLocalError as e:
            print("Lips not found in frame")    
        
        #print("Writing to path: "+f"C:\\Users\\Jai K\\CS Stuff\\cropped_frames\\frame{frameNR}.jpg")
        
        cv2.imwrite(f"C:\\Users\\Jai K\\CS Stuff\\cropped_frames\\frame{frameNR}.jpg", image)
        #print("Image added!")
        frameNR += 1
    
    return "C:\\Users\\Jai K\\CS Stuff\\cropped_frames\\"



#general_crop()
#capture_split()
#crop_user()

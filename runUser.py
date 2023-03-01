import LR
from tkinter import *
from PIL import ImageTk, Image
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
import tkinter
from tkinter.ttk import Progressbar
# from tensorflow.keras.applications.resnet50 import ResNet50
warnings.filterwarnings("ignore")
input_word = ""
win  = Tk()
#win2 = Tk()
#win3 = Tk()
win.title("First screen")
#win2.title("Second screen")
#win3.title("Third screen")
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
starting_path = "C:\\Users\\Jai K\\CS Stuff\\cropped_frames"
word_pred = ""
def main():
    #win2.destroy()
    #print("Got here...")
    MAX_WIDTH = 100
    MAX_HEIGHT = 100
    x_train = []
    model = load_model(r"C:\Users\Jai K\CS Stuff\Python\ISR Project\100model.h5")
    captureSplit.capture_split()
    captureSplit.crop_user()
    p = Progressbar(win,orient=HORIZONTAL,length=200,mode="determinate",takefocus=True,maximum=100)
    p.pack()            
    for i in range(100):                
        p.step()            
        win.update()
    cropped_frames = os.listdir(starting_path)
    print(cropped_frames)
    sequence = []
    max_seq_length = 10

    for frame in cropped_frames:

        #print("Frame!!!!: ",frame)
        frame = cv2.imread(starting_path+"\\"+frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Cropped Frame",frame)
        frame = frame.astype(np.uint8)
        sequence.append(frame)

    print("Sequence Shape: ",len(sequence),len(sequence[0]))
    pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]
    sequence.extend(pad_array * (max_seq_length - len(sequence)))
    sequence = np.array(sequence)
    if len(sequence)>10:
        sequence = sequence[:10]
        print("Length of shortened sequence: ",len(sequence))
        print("Shape of Shortened Sequence: ",sequence.shape)
    #print("Sequence shape: ",sequence.shape)
    x_train.append(sequence)
    x_train = np.array(x_train)
    #print("X: Train shape: ",x_train.shape)

    ypred = model.predict(x_train)
    ypred = np.array(ypred)
    #print("Y pred: ",ypred)
    #print("Pred shape: ",ypred.shape)
    #print("argmax: ",np.argmax(ypred,axis=1))
    word_pred = words[int(np.argmax(ypred, axis=1))]
    #print("Prediction: ", word_pred)
    predLabel.configure(text="Word Predicted: "+word_pred)
    predLabel.pack(pady=30)
    b2.configure(text="Click To Record and Try Again")




def helper():
    for i in range(5):
        waithere()
        var.set(" " if i%2 else "Word Selected: "+str(input_word))
        win.update_idletasks()

def waithere():
    var = IntVar()
    var.set
    win.after(500, var.set, 1)
    win.wait_variable(var)

def show():
    dropLabel.config(text="Word Chosen: "+clicked.get())
    global input_word
    global dropButton
    input_word = clicked.get()
    drop.destroy()
    dropLabel.destroy()
    dropButton.destroy()
    label.destroy()
    flashing.pack()
    helper()
    b2.pack()





clicked = StringVar()
clicked.set("Choose here")
drop = OptionMenu(win,clicked,*words,)
drop.config(borderwidth=1,)
label=Label(win, text="Select the word you want to say. ", font=("Courier 22 bold"),fg='green',bg='black')
label.pack(pady=20)
predLabel = Label(win,text="",font="Courier 22 bold",fg='Red',bg='black')
drop.pack(pady=50)
dropButton = Button(win,text="Click to Confirm",command=show,width=20,borderwidth=5)
dropButton.pack(pady=100)
dropLabel = Label(win,text=" ",bg='black',pady=20,font='Courier 15 bold',fg='green')
dropLabel.pack()
var = StringVar()
var.set("")
win.geometry("850x500+200+200")
win.configure(background='black')
flashing = Label(win,textvariable=var,font='Courier 22 bold',fg='green',bg='black')
b2 = Button(win, text= "Click To Record",width=25,command=main,borderwidth='5')
win.mainloop()
print("input word: ",input_word)
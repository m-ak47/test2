from flask import Flask, render_template, request
from tensorflow.keras.models import Sequential, Model
from os import listdir
import tensorflow as tf


import numpy as np

#import keras.models
from tensorflow import keras
from tensorflow.keras.models import load_model
import h5py
from Bootstrap.compiler.classes.Compiler import *
#from model import *
from tensorflow.keras.models import model_from_json
import numpy
import os
#
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
#print(tf.version.VERSION)

# path = "HTML/my_model.h5"
# new_model = tf.keras.models.load_model(path)
# # load json and create model
# json_file = open('HTML/model.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("HTML/model.h5")


# model = Model.load_weights(self.filepath="HTML/my_model_weights.h5",by_name=True)
#file = load_model("HTML/my_model.h5")
# file = h5py.File("HTML/my_model.h5",'r')
# model=list(file)
# file.clear()




app = Flask(__name__)

# model = pickle.load(open('HTML/image_processing.hdf5', 'rb'))

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


# map an integer to a word
IR2 = InceptionResNetV2(weights=None, include_top=False, pooling='avg')
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'START'
    # iterate over the whole length of the sequence
    for i in range(900):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0][-100:]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        print(' ' + word, end='')
        # stop if we predict the end of the sequence
        if word == 'END':
            break
    return

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# Initialize the function to create the vocabulary
tokenizer = Tokenizer(filters='', split=" ", lower=False)
# Create the vocabulary
tokenizer.fit_on_texts([load_doc('Bootstrap/resources/bootstrap.vocab')])
# model = init()
@app.route('/',methods = ['GET', 'POST'])
def Home():
    if request.method == 'POST':
        f = request.files['img_file']
        test_image = img_to_array(load_img(f, target_size=(299, 299)))
        test_image = np.array(test_image, dtype=float)
        test_image = preprocess_input(test_image)
        test_features = IR2.predict(np.array([test_image]))
        # html=generate_desc(loaded_model, tokenizer, np.array(test_features), 100)

        return render_template('index.html',url = test_features )


    return render_template('index.html')



if __name__=="__main__":
    app.run(debug=True)

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:47:56 2023

@author: Croma
"""
#Importing required modules:
    
import tensorflow as tf
tf.config.list_physical_devices('GPU')

import keras
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
# from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Concatenate

from matplotlib import pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from keras.models import load_model
from numpy.random import randn

from PIL import Image, ImageOps
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

mse = tf.keras.losses.MeanSquaredError()

#Reading Images

image_list = []

for filename in glob.glob('C:/Users/AatishNehe/OneDrive - str.ucture GmbH/onedrive/OneDrive - str.ucture GmbH/TEAM/Aatish/Microstructure/Images/*.jpg'):
    im=Image.open(filename)
    im=ImageOps.grayscale(im)
    im=im.resize((448, 448))
    arr = np.array(im)
    image_list.append(arr)
    
print(image_list[0].shape)

trainX = np.asarray(image_list)
# plot 25 images
for i in range(25):
    plt.subplot(5, 5, 1 + i)
    plt.axis('off')
    plt.imshow(trainX[i])
plt.show()
print(trainX[1].shape)


#Reading Material Parameters as Pandas Dataframe
output = pd.read_csv('C:/Users/AatishNehe/OneDrive - str.ucture GmbH/onedrive/OneDrive - str.ucture GmbH/TEAM/Aatish/Microstructure/output_summary.csv')
output.drop(['name'],inplace=True,axis=1)
output.head()

#Normalising Data
scaler = MinMaxScaler()
model=scaler.fit(output)
output=model.transform(output)

output[[0,1,2,3,4]]


# define the standalone discriminator model
#Given an input image, the Discriminator outputs the likelihood of the image being real.
#Binary classification - true or false (1 or 0). So using sigmoid activation.
def define_discriminator(in_shape=(448,448)):
    
    # label input
    in_label = Input(shape=(7,))  #Shape 1
   
    n_nodes = in_shape[0] * in_shape[1] #256x256 = 65536. 
    li = Dense(100, activation='linear')(in_label)  #Shape = 1, 65536/2
    li = Dense(n_nodes, activation='linear')(li)  #Shape = 1, 65536
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1))(li)  #256x256x3
    
    in_image = Input(shape=in_shape) #256x256x3
    in_image = Reshape((in_shape[0], in_shape[1], 1))(in_image)
    # concat label as a channel
    model = Concatenate()([in_image, li]) #256x256x4 (4 channels, 3 for image and the other for labels)
    
    #model = Sequential()
    model = Conv2D(64, (3,3), strides=(2,2), padding='same')(model) #128x128x128
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)  #64
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)  #32
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)  #16
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)  #8
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)  #4x4x128
    model = LeakyReLU(alpha=0.2)(model)

    model = Flatten()(model) #shape of 2048
    model = Dropout(0.4)(model)
    
    out_layer = Dense(1, activation='sigmoid')(model) #shape of 1
    
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

test_discr = define_discriminator()
print(test_discr.summary())


# define the standalone generator model
#latent vector and label as inputs

def define_generator(latent_dim):
    
    # label input
    in_label = Input(shape=(7,))  #Input of dimension 1
    
    # linear multiplication
    n_nodes = 7 * 7  # To match the dimensions for concatenation later in this step.  
    li = Dense(n_nodes)(in_label) #1x64
    # reshape to additional channel
    li = Reshape((7, 7, 1))(li)
    
    # image generator input
    in_lat = Input(shape=(1,latent_dim))  #Input of dimension 100
  
    # foundation for 8x8 image
    # We will reshape input latent vector into 8x8 image as a starting point. 
    #So n_nodes for the Dense layer can be 128x8x8 so when we reshape the output 
    #it would be 8x8x128 and that can be slowly upscaled to 32x32 image for output.
    #Note that this part is same as unconditional GAN until the output layer. 
    #While defining model inputs we will combine input label and the latent input.
    n_nodes = 128 * 7 * 7

    gen = Dense(n_nodes, activation='linear')(in_lat)  #shape=8192
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen) #Shape=8x8x128
    # merge image gen and label input
    merge = Concatenate()([gen, li])  #Shape=8x8x129 (Extra channel corresponds to the label)
    # upsample to 16x16
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge) #16x16x128
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 32x32
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #32x32x128
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #64
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #128
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(gen) #256
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(gen) #512
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (8,8), activation='tanh', padding='same')(gen) #256x256x3
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model   #Model not compiled as it is not directly trained like the discriminator.

test_gen = define_generator(100)
print(test_gen.summary())

#print(test_gen.input)


# #Generator is trained via GAN combined model. 
# define the combined generator and discriminator model, for updating the generator
#Discriminator is trained separately so here only generator will be trained by keeping
#the discriminator constant. 
def define_gan(g_model, d_model):
    d_model.trainable = False  #Discriminator is trained separately. So set to not trainable.
    
    ## connect generator and discriminator...
    # first, get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input  #Latent vector size and label size
    # get image output from the generator model
    gen_output = g_model.output  #32x32x3
    
    # generator image output and corresponding input label are inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=mse, optimizer=opt)
    return model


#Function to load input examples for the model
def load_real_samples():
    # load dataset
    trainX = np.asarray(image_list)
    trainy = output
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5  #Generator uses tanh activation so rescale 
                            #original images to -1 to 1 to match the output of generator.
    return [X, trainy]


# # select real samples
# pick a batch of random real samples to train the GAN
#In fact, we will train the GAN on a half batch of real images and another 
#half batch of fake images. 
#For each real image we assign a label 1 and for fake we assign label 0. 
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset  
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels and assign to y (don't confuse this with the above labels that correspond to cifar labels)
    y = ones((n_samples, 1))  #Label=1 indicating they are real
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, labels):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, 1, latent_dim)
    # generate labels
    ix = randint(0, 921, n_samples)
    labels_input = labels[ix]
    return [z_input, labels_input]

# use the generator to generate n fake examples, with class labels
#Supply the generator, latent_dim and number of samples as input.
#Use the above latent point generator to generate latent points. 
def generate_fake_samples(generator, latent_dim, n_samples, dataset):
    # generate points in latent space
    images, labels = dataset
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, labels)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))  #Label=0 indicating they are fake
    return [images, labels_input], y

# train the generator and discriminator
#We loop through a number of epochs to train our Discriminator by first selecting
#a random batch of images from our true/real dataset.
#Then, generating a set of images using the generator. 
#Feed both set of images into the Discriminator. 
#Finally, set the loss parameters for both the real and fake images, as well as the combined loss. 
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=4, n_batch=8):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)  #the discriminator model is updated for a half batch of real samples 
                            #and a half batch of fake samples, combined a single batch. 
    losses = []
    
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
        
             # Train the discriminator on real and fake images, separately (half batch each)
            #Research showed that separate training is more effective. 
            # get randomly selected 'real' samples
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)

            # update discriminator model weights
            ##train_on_batch allows you to update weights based on a collection 
            #of samples you provide
            d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, dataset)
            # update discriminator model weights
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            
            #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss if you want to report single..
            
            # prepare points in latent space as input for the generator
            images, labels = dataset
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch, labels)
            
            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            #This is where the generator is trying to trick discriminator into believing
            #the generated image is true (hence value of 1 for y)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
             # Generator is part of combined model where it got directly linked with the discriminator
            # Train the generator with latent_dim as x and 1 as y. 
            # Again, 1 as the output as it is adversarial and if generator did a great
            #job of folling the discriminator then the output would be 1 (true)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # Print losses on this batch
            print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f' %
            (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
            
        loss = [d_loss_real, d_loss_fake, g_loss]
            
        losses.append(loss)
            
    # save the generator model
    g_model.save('trial.h5')
    return losses
  
  
#Train the GAN

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
history = train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10)


print(history[:][0])

#Visualise Losses
plt.plot(history)
plt.xlabel('No. of Epochs') 
# naming the y axis 
plt.ylabel('Loss') 
# get current axes command
#ax = plt.gca()

#ax.legend(['D_loss_real', 'D_loss_fake', 'G_loss'])
# giving a title to my graph 
plt.title('Losses') 
    
# function to show the plot 
plt.show() 




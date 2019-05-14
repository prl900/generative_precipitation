from __future__ import print_function, division

from keras.layers import concatenate, Input, Dense, Reshape, Flatten, Dropout, Concatenate, BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from data_loader import DataLoader
import numpy as np
import os

class Unet():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.img_shape = (self.img_rows, self.img_cols, 3)

        # Configure data loader
        self.dataset_name = 'unet'
        self.data_loader = DataLoader()

        # Calculate output shape of D (PatchGAN)
        optimizer = Adam(0.0002, 0.5)
        self.unet = self.get_unet()
        self.unet.compile(loss='mse', optimizer=optimizer)

    def get_unet(self, n_filters=64, im_shape=(3,256,256)):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            e = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            e = LeakyReLU(alpha=0.2)(e)
            if bn:
                e = BatchNormalization(momentum=0.8)(e)
            return e

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            d = UpSampling2D(size=2)(layer_input)
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(d)
            if dropout_rate:
                d = Dropout(dropout_rate)(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Concatenate()([d, skip_input])
            return d

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, n_filters, bn=False)
        d2 = conv2d(d1, n_filters*2)
        d3 = conv2d(d2, n_filters*4)
        d4 = conv2d(d3, n_filters*8)
        d5 = conv2d(d4, n_filters*8)
        d6 = conv2d(d5, n_filters*8)
        d7 = conv2d(d6, n_filters*8)

        # Upsampling
        u1 = deconv2d(d7, d6, n_filters*8)
        u2 = deconv2d(u1, d5, n_filters*8)
        u3 = deconv2d(u2, d4, n_filters*8)
        u4 = deconv2d(u3, d3, n_filters*4)
        u5 = deconv2d(u4, d2, n_filters*2)
        u6 = deconv2d(u5, d1, n_filters)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def get_unet_orig(self):
        concat_axis = 3
        inputs = Input(shape=(256, 256, 3))

        bn0 = BatchNormalization(axis=3)(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
        bn1 = BatchNormalization(axis=3)(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization(axis=3)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        bn3 = BatchNormalization(axis=3)(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn3)
        bn4 = BatchNormalization(axis=3)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn4)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        bn5 = BatchNormalization(axis=3)(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn5)
        bn6 = BatchNormalization(axis=3)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn6)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        bn7 = BatchNormalization(axis=3)(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn7)
        bn8 = BatchNormalization(axis=3)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(bn8)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        bn9 = BatchNormalization(axis=3)(conv5)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(bn9)
        bn10 = BatchNormalization(axis=3)(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(bn10)
        up6 = concatenate([up_conv5, conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        bn11 = BatchNormalization(axis=3)(conv6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn11)
        bn12 = BatchNormalization(axis=3)(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(bn12)
        up7 = concatenate([up_conv6, conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        bn13 = BatchNormalization(axis=3)(conv7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn13)
        bn14 = BatchNormalization(axis=3)(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(bn14)
        up8 = concatenate([up_conv7, conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        bn15 = BatchNormalization(axis=3)(conv8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn15)
        bn16 = BatchNormalization(axis=3)(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(bn16)
        up9 = concatenate([up_conv8, conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        bn17 = BatchNormalization(axis=3)(conv9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn17)
        bn18 = BatchNormalization(axis=3)(conv9)

        conv10 = Conv2D(3, (1, 1))(bn18)
        #bn19 = BatchNormalization(axis=3)(conv10)

        return Model(inputs=inputs, outputs=conv10)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                loss = self.unet.train_on_batch(imgs_A, imgs_B)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [loss: %f] time: %s" % (epoch, epochs, batch_i, self.data_loader.n_batches, loss, elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        print(imgs_A.shape, imgs_B.shape)
        fake_A = self.unet.predict(imgs_B)
        print(fake_A.shape)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Input', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = Unet()
    gan.train(epochs=400, batch_size=10, sample_interval=200)

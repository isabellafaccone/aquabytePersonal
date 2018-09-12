import keras.backend as K
from keras.layers import merge, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.backend import binary_crossentropy


smooth = 1e-12

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def get_unet(num_channels, img_rows, img_cols):
    
    inputs = Input((img_rows, img_cols, num_channels))
    
    ## GOING DOWN ##
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, padding='same', kernel_initializer ='he_uniform', activation = 'elu')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer ='he_uniform', activation = 'elu')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    ## GOING UP ##
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    # up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation = 'elu')(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model
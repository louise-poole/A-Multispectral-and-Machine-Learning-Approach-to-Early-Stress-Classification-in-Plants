# Louise Poole

import cv2
import numpy as np
import sys
#from sklearn import preprocessing

#from sklearn.metrics import confusion_matrix
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, LabelBinarizer
from sklearn.preprocessing import LabelEncoder
#from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
#import tensorflow

from background_marker import *

from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.svm import SVC
#from sklearn import metrics
#from sklearn.decomposition import PCA as RandomizedPCA
from pathlib import Path

#Resnet specific
import tensorflow as tf
from tensorflow import Tensor
from keras.layers import ReLU, Input, Add, AveragePooling2D

from tensorflow import keras
import keras_tuner as kt #tuning
from keras_tuner import HyperParameters
#from kerastuner.applications import HyperResNet

#GoogleNet specific
from keras.layers import MaxPool2D, concatenate, GlobalAveragePooling2D

#General
#from keras.models import Sequential, Model
#from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
#from keras.optimizers import Adam
from keras.optimizers import adam_v2
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy

#import mahotas as mh
#import pickle
import joblib
import os

import multiprocessing as mp

from keras.callbacks import TensorBoard
from time import time

#statistics
from mlxtend.evaluate import mcnemar, mcnemar_table

# ---------------------------------global variables------------------------------------------
EPOCHS=10
width=101
height=101
depth=3
n_classes=2
inputShape = (0, 0, 0)
lr_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

#--------------------------------------------------------------------------------------------


# ----------------------------------LEAF SEGMENTATION-----------------------------------------


def generate_background_marker(image):
    # The following code was adpated from work by YaredTaddese on Github (https://github.com/YaredTaddese/leaf-image-segmentation)
    """
	Generate background marker for an image

	Returns:
		tuple[0] (ndarray of an image): original image
		tuple[1] (ndarray size of an image): background marker
	"""
    original_image = image

    marker = np.full((original_image.shape[0], original_image.shape[1]), True)

    # update marker based on vegetation color index technique
    color_index_marker(index_diff(original_image), marker)

    return original_image, marker


def segment_leaf(image):
    """
	Segments leaf from an image

	Returns:
		tuple[0] (ndarray): original image to be segmented
		tuple[1] (ndarray): A mask to indicate where leaf is in the image
							or the segmented image based on marker_intensity value
	"""
    # get background marker and original image
    original, marker = generate_background_marker(image)

    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)

    # FLOODFILL MARKERS
    # set up components
    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(bin_image, connectivity=8, ltype=cv2.CV_32S)

    # fill holes using opencv floodfill function
    # set up seedpoint(starting point) for floodfill
    bkg_locs = np.where(img_labeled == 0)
    bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])

    # copied image to be floodfill
    img_floodfill = bin_image.copy()

    # create a mask to ignore what shouldn't be filled
    h_, w_ = bin_image.shape
    mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)

    cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed,
                  newVal=255)
    holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.

    # get a mask to avoid filling non-holes that are adjacent to image edge
    non_holes_mask = generate_floodfill_mask(bin_image)
    holes_mask = np.bitwise_and(holes_mask, np.bitwise_not(non_holes_mask))

    bin_image = bin_image + holes_mask

    # WATERSHED
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    image = original.copy()
    waterMarkers = cv2.watershed(image, markers)

    # image[waterMarkers == -1] = [255,0,0]
    # cv2.imshow('watershed', image)

    image = original.copy()
    image[bin_image == 0] = np.array([0, 0, 0])

    return original, image


# ---------------------------------------------------------------------------------------------


# ---------------------------------DISEASE SEGMENTATION----------------------------------------
# inspired by code done by Shikhar Johri (https://github.com/johri002/Automatic-leaf-infection-identifier)

def segment_disease(image):
    img = image.copy()
    original = image.copy()

    # Calculating number of pixels with shade of white(p) to check if exclusion of these pixels is
    # required or not (if more than a fixed %) in order to differentiate the white background or
    # white patches in image caused by flash, if present.
    p = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): # if each R,G,B value is above 110, the pixel is considered 'near white'
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            if (B > 110 and G > 110 and R > 110):
                p += 1

    # finding the % of pixels in shade of white
    totalpixels = img.shape[0] * img.shape[1]
    per_white = 100 * p / totalpixels

    # excluding all the pixels with colour close to white if they are more than 10% in the image
    if per_white > 10:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):  # replace 'near white' pixels with a light grey (200, 200, 200)
                B = img[i][j][0]
                G = img[i][j][1]
                R = img[i][j][2]
                if (B > 110 and G > 110 and R > 110):
                    img[i][j] = [200, 200, 200]
    # cv2.imshow('color change', img)

    # Guassian Blur
    blur1 = cv2.GaussianBlur(img, (3, 3), 1)

    # Mean-Shift
    newimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)

    # Guassian Blur
    blur = cv2.GaussianBlur(img, (11, 11), 1)

    # Changing Colour-Space to HSL
    imghls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    imghls[np.where((imghls == [30, 200, 2]).all(axis=2))] = [0, 200, 0]

    # Only Hue Channel
    huehls = imghls[:, :, 0]
    huehls[np.where(huehls == [0])] = [35]

    # Threshold Hue Image
    ret, thresh = cv2.threshold(huehls, 28, 255, cv2.THRESH_BINARY_INV)

    # Mask Thresholded image from original image
    mask = cv2.bitwise_and(original, original, mask=thresh)

    # cv2.imshow('disease segment', mask)
    return mask


# ---------------------------------------------------------------------------------------------


# ----------------------------------COMBINED SEGMENTATION--------------------------------------

def segment_image(image):
    try: # need a try-catch because in images that are close ups of leaves, there is no background. So leaf isolation fails as there is nothing to isolate and no border to discover
        original, segmentedLeaf = segment_leaf(image)
        segmentedDisease = segment_disease(original)

        # Combine the Two Segmentations
        sub = cv2.subtract(segmentedLeaf,
                           segmentedDisease)  # subtract the images to prevent altering values when we add them again
        result = cv2.add(sub, segmentedDisease)

    except:
        result = image
    return result


# ---------------------------------------------------------------------------------------------


# -----------------------------LOAD AND SEGMENT IMAGE DATABASE---------------------------------

def getImagesAndTargets(folders):
    totalFiles = 0
    for i, direc in enumerate(folders):
        list = os.listdir(direc)
        number_files = len(list)
        totalFiles += number_files

    # allImages = np.zeros((totalFiles, 64, 64, 3), dtype=np.uint8)
    # allImages = np.zeros((totalFiles, 256, 256), dtype=np.uint8)
    allImages = np.zeros((totalFiles, 256, 256, 3), dtype=np.uint8)

    target = []
    count = 0
    for i, direc in enumerate(folders):
        # print(direc.name)
        images = []
        imageNum = 0

        for file in direc.iterdir():
            # read in file
            # print('Reading file ' + str(count + 1))
            img = cv2.imread(str(file)) #IF USING NORMAL IMAGES

            allImages[count] = cv2.resize(img, (256,256))
            # allImages.append(img)
            target.append(i)
            count = count + 1

    return allImages, target

def loadImageDataset(folder, prepro='-g'):
    print('Loading Dataset')
    path = Path(folder)
    folders = [directory for directory in path.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    descr = "Given image dataset"

    images, target = getImagesAndTargets(folders)
    data = []

    data = np.zeros((len(images), 256, 256, 3))
    count = 0
    for img in images:
        #print('Processing image ' + str(count + 1))
        #print('SEGMENTING')
        # segment image
        # img_segmented = segment_image(img)
        #grayImg = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
        # writeTo = os.path.join('D:\Project2019\combined\{}-SEGMENTED\image{}'.format(path,count+1), '.jpg')
        # print(writeTo)
        # cv2.imwrite(writeTo, grayImg)
        img = cv2.resize(img, (256,256))
        #print('LBP')
        # apply feature descriptor
        #lbp = mh.features.lbp(grayImg, radius=4, points=16)

        # add to database
        # flat_data.append(img_resized.flatten())
        # images.append(img_resized)

        if prepro == '-g':
            imgToAdd = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif prepro == '-s':
            imgToAdd = segment_image(img)
        else: # prepro == '-c'
            imgToAdd = img

        imgArr = img_to_array(imgToAdd)
        data[count] = np.array(imgArr, dtype=np.float16) / 225.0
        count += 1

    # flat_data = np.array(flat_data)
    target = np.array(target)

    if prepro == '-g':
        print('Loaded in greyscale')
    elif prepro == '-s':
        print('Loaded in segmented colour')
    else: # prepro == '-c'
        print("Loaded in colour")

    # return in the same format as the buit in datasets
    return Bunch(data=data, target=target, target_names=categories, DESCR=descr)

def getTargetsFromDirectory(folder):
    path = Path(folder)
    folders = [directory for directory in path.iterdir() if directory.is_dir()]
    totalFiles = 0
    for i, direc in enumerate(folders):
        print(i)
        print(direc)
        list = os.listdir(direc)
        number_files = len(list)
        totalFiles += number_files

    target = []
    count = 0
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            target.append(i)
            count = count + 1

    return target


def segmentDatabase(folder):
    path = Path(folder)
    print(str(path))
    folders = [directory for directory in path.iterdir() if directory.is_dir()]
    count = 0
    for i, direc in enumerate(folders):
        # print(direc.name)
        images = []
        imageNum = 0

        for file in direc.iterdir():
            # read in file
            print('Reading file ' + str(count + 1))

            img = cv2.imread(str(file))
            segmentedImage = segment_image(img)
            # print('{}\{}\SEGMENTED{}.jpg'.format(folder, direc.name, str(count+1)))
            cv2.imwrite(
                '{}\{}\SEGMENTED{}.jpg'.format(folder, direc.name, str(count + 1)),
                segmentedImage)
            # images.append(img)
            count = count + 1

def loadAlreadySegmented(folder):
    print('Loading Segmented Dataset')
    path = Path(folder)
    folders = [directory for directory in path.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    descr = "Given image dataset"

    images, target = getImagesAndTargets(folders)
    data = []
    data = np.zeros((len(images), 4116))
    count = 0

    for img in images:
        # apply feature descriptor
        lbp = mh.features.lbp(img, radius=4, points=16)

        # add to database
        data[count] = lbp
        count += 1

    # 	DB free ram
    del images
    target = np.array(target)
    # return in the same format as the buit in datasets
    return Bunch(data=data, target=target, target_names=categories, DESCR=descr)

# ---------------------------------------------------------------------------------------------


# -------------------------------------ARCHITECTURES-------------------------------------------

def oluwafemi(inputShape, n_classes):
    model = keras.Sequential()
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
    return model

def resNet(inputShape, n_classes):
    def reluBatchNormalization(inputs: Tensor) -> Tensor:
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn

    def residualBlock(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
        y = Conv2D(kernel_size=kernel_size,
                strides= (1 if not downsample else 2),
                filters=filters,
                padding="same")(x)
        y = reluBatchNormalization(y)
        y = Conv2D(kernel_size=kernel_size,
                strides=1,
                filters=filters,
                padding="same")(y)
        if downsample:
            x = Conv2D(kernel_size=1,
                    strides=2,
                    filters=filters,
                    padding="same")(x)
        out = Add()([x, y])
        out = reluBatchNormalization(out)
        return out

    inputs = Input(shape=inputShape)
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = reluBatchNormalization(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residualBlock(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(n_classes, activation='softmax')(t)

    model = keras.Model(inputs, outputs)

    return model

def googleNet(inputShape, n_classes):#https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
    def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
        conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

        conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

        conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

        pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

        return output

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    input_layer = Input(shape=inputShape)

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32, name='inception_3a')
    x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64, name='inception_3b')
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    x = inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64, name='inception_4a')

    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(n_classes, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64, name='inception_4b')
    x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64, name='inception_4c')
    x = inception_module(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64, name='inception_4d')

    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(n_classes, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, name='inception_4e')
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, name='inception_5a')
    x = inception_module(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128, name='inception_5b')

    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    x = Dropout(0.4)(x)
    x = Dense(n_classes, activation='softmax', name='output')(x)#was

    model = Model(input_layer, [x, x1, x2], name='inception_v1')

    return model

def VGG(inputShape, n_classes):#https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c and https://github.com/simongeek/KerasVGGcifar10
    model = Sequential()

    #ConvBlock2
    model.add(Conv2D(input_shape=inputShape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #ConvBlock2
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #ConvBlock3
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #ConvBlock3
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #ConvBlock3
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=n_classes, activation="softmax"))

    return model

# ---------------------------------------------------------------------------------------------


# ----------------------------------------TUNING-----------------------------------------------

def buildOluwafemi(hp):
    inputShape = (height, width, depth)
    print(n_classes)

    model = keras.Sequential()
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
    model.add(keras.layers.experimental.preprocessing.RandomRotation(0.2))
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    #hyperparameters
    hp_opt = hp.Choice('optimizer', values=['Adam','SGD'])
    hp_lr = hp.Choice('learning_rate', values=lr_values)
    if hp_opt == 'Adam':
        #optimiser
        opt = adam_v2.Adam(learning_rate=hp_lr, decay=hp_lr / EPOCHS)
    else:
        opt = keras.optimizers.SGD(learning_rate=hp_lr)

    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    return model

def buildModdedOluwafemi(hp):


    model = keras.Sequential()
    inputs = Input(shape=inputShape)
    x = inputs

    x = keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(x)
    x = keras.layers.experimental.preprocessing.RandomRotation(0.2)(x)
    #add translation

    numBlocks = hp.Int('numConvBlocks', min_value=1, max_value=5, step=1)

    for i in range(numBlocks):
        x = keras.layers.Conv2D(
            filters=hp.Int('conv_filter1.'+str(i), min_value=32, max_value=256, step=64),
            kernel_size=hp.Choice('conv_kernel1.'+str(i), values = [3,5]),
            padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if hp.Choice('pooling_'+str(i), ['avg','max']) == 'max':
            x = MaxPool2D()(x)
        else:
            x = AveragePooling2D()(x)
        x = Dropout(hp.Float('dropout_'+str(i), min_value=0.2, max_value=0.6, step=0.1))(x)
        if i<numBlocks-1:
            x = keras.layers.Conv2D(
                filters=hp.Int('conv_filter2.'+str(i), min_value=32, max_value=256, step=64),
                kernel_size=hp.Choice('conv_kernel2.'+str(i), values = [3,5]),
                padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
    x = Flatten()(x)
    x = Dense(hp.Int('hidden_size', min_value=128, max_value=256, step=64), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout_final', min_value=0.2, max_value=0.6, step=0.1))(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    #hyperparameters
    hp_lr = hp.Choice('learning_rate', values=lr_values)
    #optimiser
    opt = adam_v2.Adam(lr=hp_lr, decay=hp_lr / EPOCHS)

    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    return model

def buildResNet(hp):
    inputShape = (height, width, depth)
    def reluBatchNormalization(inputs: Tensor) -> Tensor:
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn

    def residualBlock(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
        y = Conv2D(kernel_size=kernel_size,
                strides= (1 if not downsample else 2),
                filters=filters,
                padding="same")(x)
        y = reluBatchNormalization(y)
        y = Conv2D(kernel_size=kernel_size,
                strides=1,
                filters=filters,
                padding="same")(y)
        if downsample:
            x = Conv2D(kernel_size=1,
                    strides=2,
                    filters=filters,
                    padding="same")(x)
        out = Add()([x, y])
        out = reluBatchNormalization(out)
        return out

    inputs = Input(shape=inputShape)
    num_filters = 64

    t = keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(inputs)
    t = keras.layers.experimental.preprocessing.RandomRotation(0.2)(t)


    t = BatchNormalization()(t)
    t = Conv2D(kernel_size=7,#3
               strides=2,#1
               filters=num_filters,
               padding="same")(t)
    t = reluBatchNormalization(t)
    t = MaxPooling2D(3, strides=2)(t)#this gone

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residualBlock(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(n_classes, activation='softmax')(t)

    model = keras.Model(inputs, outputs)

    #hyperparameters
    hp_opt = hp.Choice('optimizer', values=['Adam','SGD'])
    hp_lr = hp.Choice('learning_rate', values=lr_values)
    if hp_opt == 'Adam':
        #optimiser
        opt = adam_v2.Adam(learning_rate=hp_lr, decay=hp_lr / EPOCHS)
    else:
        opt = keras.optimizers.SGD(learning_rate=hp_lr)

    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    return model

def buildModdedResNet(hp):
    inputShape = (height, width, depth)
    def reluBatchNormalization(inputs: Tensor) -> Tensor:
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn

    def residualBlock(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
        y = Conv2D(kernel_size=kernel_size,
                strides= (1 if not downsample else 2),
                filters=filters,
                padding="same")(x)
        y = reluBatchNormalization(y)
        y = Conv2D(kernel_size=kernel_size,
                strides=1,
                filters=filters,
                padding="same")(y)
        if downsample:
            x = Conv2D(kernel_size=1,
                    strides=2,
                    filters=filters,
                    padding="same")(x)
        out = Add()([x, y])
        out = reluBatchNormalization(out)
        return out

    inputs = Input(shape=inputShape)
    num_filters = hp.Int('num_filters', min_value=32, max_value=128, step=32)

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=hp.Choice('kernel_size_Conv1', values = [3,5]),
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = reluBatchNormalization(t)

    nb1 = hp.Int('num_blocks_1', min_value=1, max_value=6, step=1)
    nb2 = hp.Int('num_blocks_2', min_value=1, max_value=6, step=1)
    nb3 = hp.Int('num_blocks_3', min_value=1, max_value=6, step=1)
    nb4 = hp.Int('num_blocks_4', min_value=1, max_value=6, step=1)
    num_blocks_list = [nb1, nb2, nb3, nb4]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residualBlock(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(n_classes, activation='softmax')(t)

    model = keras.Model(inputs, outputs)

    #hyperparameters
    hp_lr = hp.Choice('learning_rate', values=lr_values)
    #optimiser
    opt = adam_v2.Adam(lr=hp_lr, decay=hp_lr / EPOCHS)

    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    return model

def buildGoogleNet(hp):#https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
    inputShape = (height, width, depth)

    def inception_block(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
        conv_1x1 = Conv2D(filters=filters_1x1, 
        kernel_size=(1, 1), 
        padding='same', 
        activation='relu', 
        kernel_initializer=kernel_init, 
        bias_initializer=bias_init)(x)


        conv_3x3 = Conv2D(filters=filters_3x3_reduce, 
        kernel_size=(1, 1), 
        padding='same', 
        activation='relu', 
        kernel_initializer=kernel_init, 
        bias_initializer=bias_init)(x)

        conv_3x3 = Conv2D(filters=filters_3x3, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu', 
        kernel_initializer=kernel_init, 
        bias_initializer=bias_init)(conv_3x3)


        conv_5x5 = Conv2D(filters=filters_5x5_reduce, 
        kernel_size=(1, 1), 
        padding='same', 
        activation='relu', 
        kernel_initializer=kernel_init, 
        bias_initializer=bias_init)(x)

        conv_5x5 = Conv2D(filters=filters_5x5, 
        kernel_size=(5, 5), 
        padding='same', 
        activation='relu', 
        kernel_initializer=kernel_init, 
        bias_initializer=bias_init)(conv_5x5)
        

        pool_proj = MaxPool2D((3, 3), 
        strides=(1, 1), 
        padding='same')(x)

        pool_proj = Conv2D(filters=filters_pool_proj, 
        kernel_size=(1, 1), 
        padding='same', 
        activation='relu', 
        kernel_initializer=kernel_init, 
        bias_initializer=bias_init)(pool_proj)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

        return output

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    input_layer = Input(shape=inputShape)

    x = keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(input_layer)
    x = keras.layers.experimental.preprocessing.RandomRotation(0.2)(x)

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_block(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32, name='inception_3a')
    x = inception_block(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64, name='inception_3b')
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    x = inception_block(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64, name='inception_4a')

    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(n_classes, activation='softmax', name='auxilliary_output_1')(x1)
    #x1 = Flatten()(x1)

    x = inception_block(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64, name='inception_4b')
    x = inception_block(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64, name='inception_4c')
    x = inception_block(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64, name='inception_4d')

    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(n_classes, activation='softmax', name='auxilliary_output_2')(x2)
    #x2 = Flatten()(x2)

    x = inception_block(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, name='inception_4e')
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    x = inception_block(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, name='inception_5a')
    x = inception_block(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128, name='inception_5b')

    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    x = Dropout(0.4)(x)
    x = Dense(n_classes, activation='softmax', name='output')(x)
    #x = Flatten()(x)

    model = keras.Model(input_layer, [x, x1, x2], name='inception_v1')

    #hyperparameters
    hp_opt = hp.Choice('optimizer', values=['Adam','SGD'])
    hp_lr = hp.Choice('learning_rate', values=lr_values)
    if hp_opt == 'Adam':
        #optimiser
        opt = adam_v2.Adam(learning_rate=hp_lr, decay=hp_lr / EPOCHS)
    else:
        opt = keras.optimizers.SGD(learning_rate=hp_lr)

    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])#categorical!!!!!!!!!!!!!!!!!!!!!!!!!

    return model

def buildVGG(hp):#https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c and https://github.com/simongeek/KerasVGGcifar10
    inputShape = (height, width, depth)
    model = keras.Sequential()

    model.add(keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
    model.add(keras.layers.experimental.preprocessing.RandomRotation(0.2))

    #ConvBlock2
    model.add(Conv2D(input_shape=inputShape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #ConvBlock2
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #ConvBlock3
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #ConvBlock3
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    #ConvBlock3
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=n_classes, activation="softmax"))

    #hyperparameters
    hp_opt = hp.Choice('optimizer', values=['Adam','SGD'])
    hp_lr = hp.Choice('learning_rate', values=lr_values)
    if hp_opt == 'Adam':
        #optimiser
        opt = adam_v2.Adam(learning_rate=hp_lr, decay=hp_lr / EPOCHS)
    else:
        opt = keras.optimizers.SGD(learning_rate=hp_lr)

    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    return model

#----------------------------------------------------------------------------------------------


# -------------------------------------TRAINING------------------------------------------------

# custom generator for GoogleNet
def multiple_outputs(generator, x_train, y_train, BS, subset1):
    gen = generator.flow(x_train,y_train,batch_size=BS,subset=subset1)
    batches = 0

    while True:
        gnext = gen.next()
        # return image batch and 3 sets of lables
        yield gnext[0], [gnext[1], gnext[1], gnext[1]]

        batches+=1
        if batches>=len(x_train)/BS:
            break

def multiple_outputs_directory(generator, directory, BS, subset):
    gen = generator.flow_from_directory(
        directory+subset,
        target_size=(256,256),
        batch_size=BS,
        class_mode='binary',
        seed=23
        ) 
        
    batches = 0

    while True:
        gnext = gen.next()
        # return image batch and 3 sets of lables
        yield gnext[0], [gnext[1], gnext[1], gnext[1]]

        batches+=1
        if batches>=len(gen.n)/BS:
            break

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1_val

def trainModel(dataset, filename, target_names, archi=1):
    print('Training phase...')

    #training parameters
    EPOCHS = 10
    INIT_LR = 1e-4
    BS = 8
    width=256
    height=256
    depth=3
    n_classes = len(dataset.target_names)
    inputShape = (height, width, depth)
    testSize = 0.5

    print(f'Learning rate: {INIT_LR}')
    print(f'Batch Size: {BS}')

    #prepare dataset
    x = dataset.data
    #y = to_categorical(dataset.target, n_classes)
    Y = dataset.target
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    y = to_categorical(encoded_Y) #one hot encoded

    #train:test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize, random_state=23)
    original_y_test = y_test

    del x, y #rescuing precious memory

    #augment the dataset
    aug = ImageDataGenerator(
        rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2,
        zoom_range=0.2,horizontal_flip=True,
        fill_mode="nearest")

    # create architecture
    model = keras.Sequential()
    if archi == 1:
        # Oluwafemi Tairu's Architecture for leaf disease detection
        model = oluwafemi(inputShape, n_classes)
    elif archi == 2:
        #ResNet
        model = resNet(inputShape, n_classes)
    elif archi == 3:
        #GoogleNet
        model = googleNet(inputShape, n_classes)
        y_train = [y_train, y_train, y_train]
        y_test = [y_test, y_test, y_test]
        # #augment the dataset
        # generator = ImageDataGenerator(...)
        # def generate_data_generator(generator, X, Y1, Y2):
        #     genX1 = generator.flow(X, Y1, seed=7)
        #     genX2 = generator.flow(X, Y2, seed=7)
        #     while True:
        #         X1i = genX1.next()
        #         X2i = genX2.next()
        #         yield X1i[0], [X1i[1], X2i[1]]
    elif archi == 4:
        #VGG
        model = VGG(inputShape, n_classes)

    #optimiser
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    #model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=[f1])
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    if archi == 3:
        history = model.fit(
            x=x_train, y=y_train,
            validation_data=(x_test, y_test),
            batch_size=BS,
            epochs=EPOCHS, verbose=1,
            )#validation_steps=len(x_test) // BS,
            #steps_per_epoch=len(x_train) // BS,
        accuracy = 'output_accuracy'
        print('Data Not Augmmented')
    else:
        history = model.fit_generator(
            aug.flow(x_train, y_train, batch_size=BS),
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // BS,
            epochs=EPOCHS, verbose=1
            )
        accuracy = 'accuracy'
        print('Data Augmented')

    del x_train, y_train #rescuing precious memory

    model.summary()
    #display accuracy information
    scores = model.evaluate(x_test, y_test)
    if archi == 3:
        print(f"Test Accuracy: {scores[4]*100}")
    else:
        print(f"Test Accuracy: {scores[1]*100}")
    maxAcc = max(history.history[f'val_{accuracy}'])
    print(f"Max Accuracy: {maxAcc*100}")
    print(f"At epoch: {history.history[f'val_{accuracy}'].index(maxAcc)+1}")

    #plot train and val accuracy and loss curves
    plotCurves(history, archi, accuracy)

    #confusion matrix
    if archi != 3:
        if archi == 1:
            y_pred = model.predict_classes(x_test)
        else:
            y_pred = np.argmax(model.predict(x_test), axis=1)
        y_test = np.argmax(y_test, axis=1)
        confusionMatrix(y_test, y_pred)

    # save the model to disk
    model.classes_ = np.array(dataset.target_names)
    joblib.dump(model, filename)

def gridSearchModel(dataset, filename, target_names, archi=1):
    print(f'ARCHITECTURE {archi}')
    print('GridSearch phase...')
    global EPOCHS
    global width
    global height
    global depth
    global n_classes
    global inputShape
    #training parameters
    EPOCHS = 1000
    #INIT_LR = 1e-4
    AUG=True
    BS = 32
    width=256
    height=256
    depth=3
    n_classes = len(dataset.target_names)
    inputShape = (height, width, depth)
    testSize = 0.2
    valSize = 0.1

    if AUG: print('DATA AUGMENTED')

    #prepare dataset
    x = dataset.data
    #y = to_categorical(dataset.target, n_classes)
    Y = dataset.target
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    y = to_categorical(encoded_Y) #one hot encoded

    # Standardization
    # scaler = preprocessing.StandardScaler().fit(x)
    # x = scaler.transform(x)

    # Train:Test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize, random_state=32)
    #original_y_test = y_test
    # print(f'xtrain size: {len(x_train)}')
    # print(f'ytrain size: {len(y_train)}')
    # x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size=valSize)
    # x_train = np.concatenate((x_train1, x_val))
    # y_train = np.concatenate((y_train1, y_val))

    x_train,y_train = shuffle(x_train,y_train) # shuffle the training data
    del x, y #rescuing precious memory

    if AUG:
        #augment the dataset
        aug = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest",
            brightness_range=[0.2,1.0],
            validation_split=valSize
            #rescale=1.0/255.0
            )
        #aug.fit(x_train) # calculate scaling statistic on the training set
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=valSize, random_state=23) #split for validation data to avoid it getting augmented
        train_generator = aug.flow(x_train,y_train, batch_size=BS,shuffle=True,subset='training',save_to_dir='Augmented',save_prefix='augment',save_format='jpg') #augment train data
        val_generator = aug.flow(x_train,y_train,batch_size=BS,shuffle=True,subset='validation')

    hp = HyperParameters()

    accuracy = 'accuracy'
    valAccuracy = 'val_accuracy'
    # create architecture
    modelBuilder = buildOluwafemi # Oluwafemi Tairu's Architecture for leaf disease detection
    if archi == 2:
        #ResNet
        modelBuilder = buildResNet
    elif archi == 3:
        #GoogleNet
        modelBuilder = buildGoogleNet
        if AUG:
            train_generator = multiple_outputs(aug,x_train,y_train,BS,'training')
            val_generator = multiple_outputs(aug,x_train,y_train,BS,'validation')
        else:
            y_train = [y_train, y_train, y_train]
        y_test = [y_test, y_test, y_test]
        accuracy = 'output_accuracy'
        valAccuracy = 'val_output_accuracy'

    elif archi == 4:
        #VGG
        modelBuilder = buildVGG
    elif archi == 5:
        #modded Oluwafemi
        modelBuilder = buildModdedOluwafemi
    elif archi == 6:
        #built in ResNet152V2
        modelBuilder = buildResNet152V2

    #tuner = RandomSearch(
    #TuningModel(num_classes=NO_CLASSES, input_shape=INPUT_SHAPE).tune_minivgg,
    #hyperparameters=hp,
    #objective='val_accuracy',
    #max_trials=TRIALS,  # more is best, time permitting
    #seed=rseed,
    ## max_epochs=EPOCHS, #for Hyperband (not BayesianOptimization)
    #executions_per_trial=1,  # if you want a more stable model, but LR is more important imho
    #directory='/home/vorse/tuning_dir',
    #project_name='palmprint_models',
    #overwrite=True,  # Disallows continuation (add more trials) alternatively manually del tuning_dir
    #tune_new_entries=False,  # confusing to say the least, think it tunes even if no pars specified
    #allow_new_entries=False,
    ## save_weights_only=True #does this work?


    # tuning
    #tuner = kt.Hyperband(
    # modelBuilder, 
    # objective=kt.Objective(valAccuracy, direction='max'), 
    # max_epochs=EPOCHS, 
    # max_trials=10, 
    # factor=3, 
    # directory='tuning', 
    # project_name='resNet', 
    # overwrite=True
    # )

    tuner = kt.BayesianOptimization(
        hypermodel=modelBuilder,
        hyperparameters=hp,
        objective=kt.Objective(valAccuracy, direction='max'),
        max_trials=25,
        overwrite=True
    )
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
    tuner.search(x_train, y_train, epochs=25, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best learning rate: {best_hps.get('learning_rate')}")
    print(f"Best optimizer: {best_hps.get('optimizer')}")
    #print(f"Best batch size: {best_hps.get('batch_size')}")

    # Build model with optimal hyperparameters train it to find best epoch
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2)

    val_acc_per_epoch = history.history[valAccuracy]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print(f'Best epoch: {best_epoch}')

    #train optial model and test
    bestModel = tuner.hypermodel.build(best_hps)
    bestModel.fit(x_train, y_train, epochs=best_epoch)
    result1 = bestModel.evaluate(x_train, y_train)
    result2 = bestModel.evaluate(x_test, y_test)
    bestModel.summary()

    # Build optimal model
    bestModel = tuner.hypermodel.build(best_hps)
    #history = bestModel.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2, callbacks=[stop_early])
    if AUG:
        #stop_early = CustomStopper(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min'))
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')
        history = bestModel.fit(train_generator, epochs=EPOCHS, steps_per_epoch=len(x_train)//BS, validation_data=val_generator, callbacks=[stop_early])
    else:
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')
        history = bestModel.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2, callbacks=[stop_early])

    # Evaluate and display
    result1 = bestModel.evaluate(x_train, y_train)
    result2 = bestModel.evaluate(x_test, y_test)
    bestModel.summary()

    print(f"Best learning rate: {best_hps.get('learning_rate')}")
    print(f"Best optimizer: {best_hps.get('optimizer')}")
    print(bestModel.metrics_names)
    print("[train loss, train accuracy]:", result1)
    print("[test loss, test accuracy]:", result2)


    #plot train and val accuracy and loss curves
    plotCurves(history, archi, accuracy)

    #confusion matrix
    if archi != 3:
        if archi == 1:
            y_pred = bestModel.predict_classes(x_test)
        else:
            y_pred = np.argmax(bestModel.predict(x_test), axis=1)
        y_test = np.argmax(y_test, axis=1)
        confusionMatrix(y_test, y_pred)
       
    # save the model to disk
    bestModel.classes_ = np.array(dataset.target_names)
    bestModel.save(filename)
    #joblib.dump(bestModel, filename)

def gridSearchModelFromDirectory(directory, filename, archi=1):
    print(f'ARCHITECTURE {archi}')
    print('GridSearch phase...')
    global EPOCHS
    global width
    global height
    global depth
    global n_classes
    global inputShape

    #training parameters
    EPOCHS = 600 #600
    BS = 8#32
    width=256
    height=256
    depth=3
    inputShape = (height, width, depth)
    #testSize = 0.2
    valSize = 0.1

    print('DATA AUGMENTED')

    #augment the dataset
    aug = ImageDataGenerator(
        #rescale=1./255,#loss stuck around 0.67
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        brightness_range=[0.2,1.0],
        validation_split=valSize
        )

    # Augment data
    print('Training data')
    train_generator = aug.flow_from_directory(
        directory+'/train',
        target_size=(256,256),
        batch_size=BS,
        class_mode='categorical',#ALL were categorical
        seed=23
        ) 

    notAug = ImageDataGenerator()
    print('Validation data')
    val_generator = notAug.flow_from_directory(
        directory+'/validation',
        target_size=(256,256),
        batch_size=BS,
        class_mode='categorical',
        seed=23
        )

    #Import test set
    print('Testing data')
    test_generator = notAug.flow_from_directory(
        directory+'/test',
        target_size=(256,256),
        shuffle=False,
        class_mode='categorical'
    )

    # Get x_train and y_train from train_generator
    total_images = train_generator.n  
    steps = (total_images//BS) + 1
    #iterations to cover all data
    x , y = [] , []
    for i in range(steps):
        temp_a , temp_b = train_generator.next()
        x.extend(temp_a) 
        y.extend(temp_b)
    x_train=np.array(x)
    y_train=np.array(y)
    
    # Get x_test and y_test from test_generator
    total_images2 = test_generator.n  
    steps = (total_images2//BS) + 1
    #iterations to cover all data
    x , y = [] , []
    for i in range(steps):
        temp_a , temp_b = test_generator.next()
        x.extend(temp_a) 
        y.extend(temp_b)
    x_test=np.array(x)
    y_test=np.array(y)

    # Get x_val and y_val from val_generator
    total_images3 = val_generator.n  
    steps = (total_images3//BS) + 1
    #iterations to cover all data
    x , y = [] , []
    for i in range(steps):
        temp_a , temp_b = val_generator.next()
        x.extend(temp_a) 
        y.extend(temp_b)
    x_val=np.array(x)
    y_val=np.array(y)

    print(f'\nTest size: {len(x_test)}')
    print(f'Train size: {len(x_train)}')

    n_classes = len(train_generator.class_indices)

    hp = HyperParameters()

    accuracy = 'accuracy'
    valAccuracy = 'val_accuracy'
    monitorObjective = 'val_loss'
    # create architecture
    modelBuilder = buildOluwafemi # Oluwafemi Tairu's Architecture for leaf disease detection
    if archi == 2:
        #ResNet
        modelBuilder = buildResNet
    elif archi == 3:
        #GoogleNet
        modelBuilder = buildGoogleNet
        train_generator = multiple_outputs_directory(aug,directory+'/',BS,'train')
        val_generator = multiple_outputs_directory(notAug,directory+'/',BS,'validation')
        test_generator = multiple_outputs_directory(notAug,directory+'/',BS,'test')
        y_train = [y_train, y_train, y_train]
        y_test = [y_test, y_test, y_test]
        y_val = [y_val, y_val, y_val]
        accuracy = 'output_accuracy'
        valAccuracy = 'val_output_accuracy'
        monitorObjective = 'loss'
    elif archi == 4:
        #VGG
        modelBuilder = buildVGG
    elif archi == 5:
        #built in modded ResNet
        modelBuilder = buildModdedResNet

    # Tuning
    tuner = kt.BayesianOptimization(
        hypermodel=modelBuilder,
        hyperparameters=hp,
        objective=kt.Objective(monitorObjective, direction='min'),
        max_trials=20,##############################20
        overwrite=True
        )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=monitorObjective, 
        mode='min',
        patience=20
        )
    
    print('SEARCHING')

    tuner.search(
        x_train, 
        y_train,
        epochs=100,#########################100
        validation_split=0.2,
        callbacks=[stop_early],
        verbose=2,
        batch_size=BS
        )
    print('Done searching')
    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Best learning rate: {best_hps.get('learning_rate')}")
    print(f"Best optimizer: {best_hps.get('optimizer')}")

    # Build optimal model
    bestModel = tuner.hypermodel.build(best_hps)
    print(f'BEST HYPERPARAMETERS{best_hps}')

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=monitorObjective, 
        patience=250, 
        restore_best_weights=True, 
        mode='min'
        )

    print('BEGIN FITTING')
    if(archi == 3):
        history = bestModel.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            validation_data=(x_val,y_val),
            verbose=2
        )
    else:
        history = bestModel.fit(
            train_generator, 
            epochs=EPOCHS, 
            #steps_per_epoch=len(x_train)//BS, #TOOK OUT FOR GOOGLENET
            validation_data=val_generator, 
            callbacks=[stop_early],
            #callbacks=[tensorboard],
            verbose=2
            )

    # Evaluate and display
    result1 = bestModel.evaluate(x_train, y_train)
    result2 = bestModel.evaluate(x_test, y_test)
    result3 = bestModel.evaluate(x_val, y_val)
    bestModel.summary()

    val_acc_per_epoch = history.history[valAccuracy]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print(f'Best epoch: {best_epoch}')

    print(f"Best learning rate: {best_hps.get('learning_rate')}")
    print(f"Best optimizer: {best_hps.get('optimizer')}")
    print(bestModel.metrics_names)
    print(f'Tested on {len(x_test)} samples')
    print("[train loss, train accuracy]:", result1)
    print("[test loss, test accuracy]:", result2)
    print("[val loss, val accuracy]", result3)
    if(archi!=3):
        print("Train F1score", get_f1(y_train,bestModel.predict(x_train)))
        print("Test F1score", get_f1(y_test,bestModel.predict(x_test)))

    #print(train_generator.class_indices.keys())

    #plot train and val accuracy and loss curves
    plotCurves(history, archi, accuracy)

    #confusion matrix
    if archi != 3:
        if archi == 1:
            y_pred = bestModel.predict_classes(x_test)
        else:
            y_pred = np.argmax(bestModel.predict(x_test), axis=1)
        y_test = np.argmax(y_test, axis=1)
        confusionMatrix(y_test, y_pred, train_generator.class_indices.keys())
        bestModel.classes_ = np.array(train_generator.class_indices.keys())
       
    # save the model to disk
    bestModel.save(filename)
    #joblib.dump(bestModel, filename)

def evaluateModelFromDirectory(directory, filename, archi=1):
    print(f'ARCHITECTURE {archi}')
    print('GridSearch phase...')
    global EPOCHS
    global width
    global height
    global depth
    global n_classes
    global inputShape

    #training parameters
    EPOCHS = 600 #600
    BS = 32#32 or 8
    width=256
    height=256
    depth=3
    inputShape = (height, width, depth)
    #testSize = 0.2
    valSize = 0.1

    print('DATA AUGMENTED')

    #augment the dataset
    aug = ImageDataGenerator(
        #rescale=1./255,#loss stuck around 0.67
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        brightness_range=[0.2,1.0],
        validation_split=valSize
        )

    # Augment data
    print('Training data')
    train_generator = aug.flow_from_directory(
        directory+'/train',
        target_size=(256,256),
        batch_size=BS,
        class_mode='categorical',#ALL were categorical
        seed=23
        ) 

    notAug = ImageDataGenerator()
    print('Validation data')
    val_generator = notAug.flow_from_directory(
        directory+'/validation',
        target_size=(256,256),
        batch_size=BS,
        class_mode='categorical',
        seed=23
        )

    #Import test set
    print('Testing data')
    test_generator = notAug.flow_from_directory(
        directory+'/test',
        target_size=(256,256),
        shuffle=False,
        class_mode='categorical'
    )

    # Get x_train and y_train from train_generator
    total_images = train_generator.n  
    steps = (total_images//BS) + 1
    #iterations to cover all data
    x , y = [] , []
    for i in range(steps):
        temp_a , temp_b = train_generator.next()
        x.extend(temp_a) 
        y.extend(temp_b)
    x_train=np.array(x)
    y_train=np.array(y)
    
    # Get x_test and y_test from test_generator
    total_images2 = test_generator.n  
    steps = (total_images2//BS) + 1
    #iterations to cover all data
    x , y = [] , []
    for i in range(steps):
        temp_a , temp_b = test_generator.next()
        x.extend(temp_a) 
        y.extend(temp_b)
    x_test=np.array(x)
    y_test=np.array(y)

    # Get x_val and y_val from val_generator
    total_images3 = val_generator.n  
    steps = (total_images3//BS) + 1
    #iterations to cover all data
    x , y = [] , []
    for i in range(steps):
        temp_a , temp_b = val_generator.next()
        x.extend(temp_a) 
        y.extend(temp_b)
    x_val=np.array(x)
    y_val=np.array(y)

    print(f'\nTest size: {len(x_test)}')
    print(f'Train size: {len(x_train)}')

    n_classes = len(train_generator.class_indices)

    hp = HyperParameters()

    accuracy = 'accuracy'
    valAccuracy = 'val_accuracy'
    monitorObjective = 'val_loss'
    # create architecture
    modelBuilder = buildOluwafemi # Oluwafemi Tairu's Architecture for leaf disease detection
    if archi == 2:
        #ResNet
        modelBuilder = buildResNet
    elif archi == 3:
        #GoogleNet
        modelBuilder = buildGoogleNet
        train_generator = multiple_outputs_directory(aug,directory+'/',BS,'train')
        val_generator = multiple_outputs_directory(notAug,directory+'/',BS,'validation')
        test_generator = multiple_outputs_directory(notAug,directory+'/',BS,'test')
        y_train = [y_train, y_train, y_train]
        y_test = [y_test, y_test, y_test]
        y_val = [y_val, y_val, y_val]
        accuracy = 'output_accuracy'
        valAccuracy = 'val_output_accuracy'
        monitorObjective = 'loss'
    elif archi == 4:
        #VGG
        modelBuilder = buildVGG
    elif archi == 5:
        #built in modded ResNet
        modelBuilder = buildModdedResNet

    # Tuning
    tuner = kt.BayesianOptimization(
        hypermodel=modelBuilder,
        hyperparameters=hp,
        objective=kt.Objective(monitorObjective, direction='min'),
        max_trials=20,##############################20
        overwrite=True
        )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor=monitorObjective, 
        mode='min',
        patience=20
        )
    
    print('SEARCHING')

    tuner.search(
        x_train, 
        y_train,
        epochs=100,#########################100
        validation_split=0.2,
        callbacks=[stop_early],
        verbose=2,
        batch_size=BS
        )
    print('Done searching')
    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Best learning rate: {best_hps.get('learning_rate')}")
    print(f"Best optimizer: {best_hps.get('optimizer')}")

    training_acc = np.empty(10, dtype=float)
    validation_acc = np.empty(10, dtype=float)
    testing_acc = np.empty(10, dtype=float)
    training_loss = np.empty(10, dtype=float)
    validation_loss = np.empty(10, dtype=float)
    testing_loss = np.empty(10, dtype=float)

    for i in range(10):
        print('')
        print(f'REPEAT {i}')
        # Build optimal model
        bestModel = tuner.hypermodel.build(best_hps)

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor=monitorObjective, 
            patience=250, 
            restore_best_weights=True, 
            mode='min'
            )

        print('Begin fitting')
        if(archi == 3):
            history = bestModel.fit(
                x_train,
                y_train,
                epochs=EPOCHS,
                validation_data=(x_val,y_val),
                verbose=2
            )
        else:
            history = bestModel.fit(
                train_generator, 
                epochs=EPOCHS, 
                #steps_per_epoch=len(x_train)//BS, #TOOK OUT FOR GOOGLENET
                validation_data=val_generator, 
                callbacks=[stop_early],
                #callbacks=[tensorboard],
                verbose=2
                )

        # Evaluate and display
        result1 = bestModel.evaluate(x_train, y_train, verbose=0)
        result2 = bestModel.evaluate(x_test, y_test, verbose=0)
        result3 = bestModel.evaluate(x_val, y_val, verbose=0)
        if archi == 3:
            training_loss[i], ret1, ret2, ret3, training_acc[i], ret4, ret5 = result1
            validation_loss[i], ret1, ret2, ret3, validation_acc[i], ret4, ret5 = result2
            testing_loss[i], ret1, ret2, ret3, testing_acc[i], ret4, ret5 = result3
        else:
            training_loss[i], training_acc[i] = result1
            validation_loss[i], validation_acc[i] = result2
            testing_loss[i], testing_acc[i] = result3

        val_acc_per_epoch = history.history[valAccuracy]
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print(f'Best epoch: {best_epoch}')
        print(f"Best learning rate: {best_hps.get('learning_rate')}")
        print(f"Best optimizer: {best_hps.get('optimizer')}")
        print(bestModel.metrics_names)
        print(f'Tested on {len(x_test)} samples')
        print("[train loss, train accuracy]:", result1)
        print("[test loss, test accuracy]:", result2)
        print("[val loss, val accuracy]", result3)
        if(archi!=3):
            print("Train F1score", get_f1(y_train,bestModel.predict(x_train)))
            print("Test F1score", get_f1(y_test,bestModel.predict(x_test)))

    print('')
    print('FINAL RESULTS:')
    print('')
    
    print('Training:')
    print(f'Accuracy: {training_acc}')
    print(f'Accuracy average: {np.average(training_acc)}')
    print(f'Accuracy deviation: {np.std(training_acc)}')
    print(f'Loss: {training_loss}')
    print(f'Loss average: {np.average(training_loss)}')
    print(f'Loss deviation: {np.std(training_loss)}')

    print('')
    print('Validation:')
    print(f'Accuracy: {validation_acc}')
    print(f'Accuracy average: {np.average(validation_acc)}')
    print(f'Accuracy deviation: {np.std(validation_acc)}')
    print(f'Loss: {validation_loss}')
    print(f'Loss average: {np.average(validation_loss)}')
    print(f'Loss deviation: {np.std(validation_loss)}')

    print('')
    print('Testing:')
    print(f'Accuracy: {testing_acc}')
    print(f'Accuracy average: {np.average(testing_acc)}')
    print(f'Accuracy deviation: {np.std(testing_acc)}')
    print(f'Loss: {testing_loss}')
    print(f'Loss average: {np.average(testing_loss)}')
    print(f'Loss deviation: {np.std(testing_loss)}')

    # save the model to disk
    bestModel.save(filename)
    #joblib.dump(bestModel, filename)


def mcNemarTest(file1, file2, folder):
    #load models
    print('LOADING MODELS')
    model1 = keras.models.load_model(file1)#joblib.load(modelFile)
    model2 = keras.models.load_model(file2)

    notAug = ImageDataGenerator()
    test_inf_generator = notAug.flow_from_directory(
        folder+'/Infrared',
        target_size=(256,256),
        shuffle=False,
        batch_size=1,
        class_mode=None
    )
    test_vis_generator = notAug.flow_from_directory(
        folder+'/Visible',
        target_size=(256,256),
        shuffle=False,
        batch_size=1,
        class_mode=None
    )

    y1 = np.array(getTargetsFromDirectory(folder+'/Infrared'))
    m1 = np.argmax(model1.predict_generator(test_inf_generator, steps=test_inf_generator.n, verbose=1), axis=1)
    m2 = np.argmax(model2.predict_generator(test_vis_generator, steps=test_vis_generator.n, verbose=1), axis=1)
    
    print(y1)
    print(m1)
    print(m2)

    table = mcnemar_table(y_target = y1,
    y_model1 = m1,
    y_model2 = m2)
    print(table)

    print('\nCorrected:')
    chi2, p = mcnemar(ary=table, corrected=True)
    print('chi-squared:', chi2)
    print('p-value:', p)

    print('\nExact:')
    chi2, p = mcnemar(ary=table, exact=True)
    print('chi-squared:', chi2)
    print('p-value:', p)



def confusionMatrix(y_test, y_pred, target_names): # CONFUSION MATRIX
    from sklearn.metrics import confusion_matrix
    # use seaborn plotting defaults
    import seaborn as sns

    print('Plotting matrix')

    plt.figure(1)
    sns.set()
    mat = confusion_matrix(y_test, y_pred)
    print("CM:               ", mat)
    # too fit it to size
    plt.subplots(figsize=(15, 10))
    # matplotlib 3.1.1 has bug

    ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d',
                cbar=False,
                xticklabels=target_names,
                yticklabels=target_names)
    # The this workaround for bug is only required for matplotlib 3.1.1 (I downgraded)
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)  # bloody bug!
    # ax.set_xlim(bottom + 0.5, top - 0.5)
    # for item in dataset.target_names.get_xticklabels():
    #     item.set_rotation(45)
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    # plt.setp(ax.get_xticklabels(), rotation=45)
    plt.setp(ax.get_yticklabels(), rotation='horizontal')
    plt.tight_layout()

    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()
    plt.savefig('confusionMatrix')#, horizontalalignment="center", verticalalignment="center")

def plotCurves(history, archi, accuracy): #PLOT TRAIN AND VAL CURVE
    acc = history.history[accuracy]
    val_acc = history.history[f'val_{accuracy}']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(2)
    #Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.savefig(str(f"accuracy{archi}"))

    plt.figure(3)
    #Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig(str(f"loss{archi}"))
    plt.show()

def generateTrainingFile(folder): # used for libsvm
    print('Loading Segmented Dataset')
    path = Path(folder)
    folders = [directory for directory in path.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    descr = "Given image dataset"

    images, target = getImagesAndTargets(folders)
    count = 0

    file = open("trainingFile.txt","w")

    for img in images:
        # apply feature descriptor
        lbp = mh.features.lbp(img, radius=4, points=16)

        line = str(target[count])
        for i in range(0,4115):
            line = "{} {}:{}".format(line, i, lbp[i])
        line = "{}\n".format(line)
        file.write(line)
        count += 1

    # 	DB free ram
    del images

    file.close()

# ---------------------------------------------------------------------------------------------


# -------------------------------------PREDICTING----------------------------------------------

def predict(image, modelFile):
    print('Press enter between images to continue')
    cv2.imshow('Input Image',img)

    # segment leaf area and display
    original, segmentedLeaf = segment_leaf(img)
    cv2.waitKey(0)
    cv2.imshow('Leaf segmentation', segmentedLeaf)

    # segment diseased area and display
    segmentedDisease = segment_disease(original)
    cv2.waitKey(0)
    cv2.imshow('Disease segmentation', segmentedDisease)

    # combine the two segmentations and display
    sub = cv2.subtract(segmentedLeaf,
                        segmentedDisease)  # subtract the images to prevent altering values when we add them again
    result = cv2.add(sub, segmentedDisease)
    cv2.waitKey(0)
    cv2.imshow('Combined Segmentation', result)

    # convert to greyscale image
    grayImg = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # apply feature descriptor
    lbp = mh.features.lbp(grayImg, radius=4, points=16)
    lbp = lbp.reshape(1, -1)

    # load the model from disk
    loaded_model = joblib.load(modelFile)

    # predict and display
    result = loaded_model.predict(lbp)
    print("\n-----------------------------------------\n")
    print(result[0])
    print("\n-----------------------------------------\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------

def doTheGPUThing():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if __name__ == '__main__':
    doTheGPUThing()
    filename = 'currentModelSave'
    code = sys.argv[1]
    text = sys.argv[2]
    if code == '-t':  # train model with given dataset
        dataset = loadImageDataset(text)
        trainModel(dataset, filename, dataset.target_names)
    elif code == '-ta': #train archi
        code2 = sys.argv[4]
        dataset = loadImageDataset(sys.argv[3], code2)
        trainModel(dataset, filename, dataset.target_names, archi=int(text))
    elif code == '-ts':  # train model with already segmented dataset
        dataset = loadAlreadySegmented(text)
        trainModel(dataset, filename, dataset.target_names)
    elif code == '-ga': #gridsearch archi
        code2 = sys.argv[4]
        dataset = loadImageDataset(sys.argv[3], code2)
        gridSearchModel(dataset, filename, dataset.target_names, archi=int(text))
    elif code == '-gad': #gridsearch archi from directory
        dir = sys.argv[3]
        print(f'Training from Directory: {dir}')
        gridSearchModelFromDirectory(dir,filename,archi=int(text))
    elif code == '-ead': #evaluate archi from directory
        dir = sys.argv[3]
        print(f'Training from Directory: {dir}')
        evaluateModelFromDirectory(dir,filename,archi=int(text))
    elif code == '-p':  # predict given image
        img = cv2.imread(str(text))
        predict(img, filename)
    elif code == '-s':  # segment and save given database
        print('Segmenting')
        segmentDatabase(text)
    elif code == '-f': #generate training file for libsvm
        generateTrainingFile(text)
    elif code == '-mcN': #run mcNemar stats test
        mcNemarTest(sys.argv[3],sys.argv[4],text)
    else:
        print('Error, invalid code entered.')
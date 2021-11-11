# vgg16 --> a fully connected convolutional network. Designed for reducing the parameters in the convolutinal layers, improve the training time saving.
# resnet --> a CNN made uo of series of residual blocks with skip connnections.
#            resnet adress the 'gradient vanishing problem' 
#            (more layer --> traning time increase and accuracy decrease, weights getting insignificant in back propogation)
# inception --> In image analysis, picking a fixed kernel size can be difficult as the size of the features can vary. 
#               Larger kernels are preferred for global features over a large area  of the image.
#               Smaller kernel are preferred for detecting area-specific features.
#               Therefore, different kernel sized are needed.
#               Inception achives with its architecture going wider instead of deeper.   
# A base library called 'Segmentation Models'. More info: https://github.com/qubvel/segmentation_models
# free for getting mask: apeer.com
# For using segmentation_models:
# !pip install keras==2.3.1
# !pip install tensorflow==2.1.0
# !pip install keras_applications==1.0.8
# !pip install image-classifiers==1.0.0
# !pip install efficientnet==1.0.0

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.utils import normalize
from keras.metrics import MeanIoU

# resize
size_x = 255
size_y = 255
# number of segmentation classes
n_classes = 4

# capture training image info as a list
train_images = []

for directory_path in glob.glob("datas/fs/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, (size_y, size_x))
        train_images.append(img)
# convert list to array
train_images = np.array(train_images)
print(train_images.shape)

# capture mask info as a list
train_masks = []
for directory_path in glob.glob("datas/fslabel/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (size_y, size_x), interpolation=cv2.INTER_NEAREST)
        train_masks.append(mask)
train_masks = np.array(train_masks)
# print(train_masks.shape)

# encode labels
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
np.unique(train_masks_encoded_original_shape)

train_mask_input = np.expand_dims(train_masks_encoded_original_shape, axis = 3)
print(train_mask_input.shape)
# separate the validation set
from sklearn.model_selection import train_test_split
X_1, X_test, y_1, y_test = train_test_split(train_images, train_mask_input, test_size=0.10, random_state= 0)
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X_1, y_1, test_size= 0.5, random_state=0)
print("Class values in the dataset are ...", np.unique(y_train))

# because of the one-hot, [0 --> 0 0 0 1], [1 --> 0 0 1 0], [2 --> 0 1 0 0], [3 --> 1 0 0 0]
from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_mask_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_mask_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

# reuse parameters in all models
n_clases = 4
activations = 'softmax'
LR = 1e-4
optim = keras.optimizers.Adam(LR)
 # in sm doc: focal loss (dice loss)> cross entropy
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
# in sematic segmentastion IOUScore and FScore > accuracy
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# models
# model1
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

# define the model
model1 = sm.Unet(BACKBONE1, encoder_weights = "imagenet", classes = n_classes, activation = activations)
model1.compile(optim, total_loss, metrics = metrics)

print(model1.summary())

history1 = model1.fit(X_train1, y_train_cat, batch_size = 8, epochs = 50, verbose = 1, validation_data = (X_test1, y_test_cat))
model1.save('resnet34backbone_50epochs.hdf5')

from keras.models import load_model
model1 = load_model("saved_models/resnet34backbone_59epochs.hdf5", compile = False)

#IOUSCORE
y_pred1 = model1.predict(X_test1)
y_pred1_argmax = np.argmax(y_pred1, axis = 3)
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:, :, :, 0], y_pred1_argmax)
print("Mean IoU = ", IOU_keras.result().numpy())

plt.imshow(train_images[0, :, :, 0], cmap = "gray")
plt.imshow(train_masks[0], cmap = "gray")







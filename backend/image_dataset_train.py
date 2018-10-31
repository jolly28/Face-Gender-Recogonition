import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from CNNModel import cnnmodel
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import glob,os


# initial parameters
epochs = 10
lr = 1e-3
batch_size = 64
img_dims = (90,90,3)

data = []
labels = []

# load image files from the dataset
image_files_men = [f for f in glob.glob("men/*.jpg", recursive=True)] 
random.seed(42)
random.shuffle(image_files_men)

image_files_women=[f for f in glob.glob("women/*.jpg", recursive=True)]
random.seed(42)
random.shuffle(image_files_women)

# create groud-truth label from the image path
for img in image_files_men:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)
    
    label= img.split('/')[0][0:3]
    if label=="men":
        labels.append(0)

for img in image_files_women:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)
    
    label= img.split('/')[0][0:5]
    if label=="women":
        labels.append(1)

r=random.random()
random.shuffle(data,lambda : r)
random.shuffle(labels,lambda : r)
    
# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3,random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

# build model
model = cnnmodel.build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],classes=2)

# compile the model
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),validation_data=(testX,testY),steps_per_epoch=len(trainX) // batch_size,epochs=epochs, verbose=1)

# save the model to disk
model.save('gender_pridiction.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('graph.jpg')

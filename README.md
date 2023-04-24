# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Load CIFAR-10 Dataset & use Image Data Generator to increse the size of dataset.

### STEP 3:
Import the VGG-19 as base model & add Dense layers to it.

### STEP 4:
Compile and fit the model.

### Step 5:
Predict for custom inputs using this model.

## PROGRAM

### Name : Shyam Kumar A
### Reg No : 212221230098
### Libraries
```
import pandas as pd
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from keras import Sequential
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras import utils
from sklearn.metrics import classification_report,confusion_matrix
```
### Load Dataset & Increse the size of it
```
(xtrain,ytrain),(xtest,ytest)=cifar10.load_data()

train_gen = ImageDataGenerator(rotation_range = 2,
                               horizontal_flip = True,
                               vertical_flip = False,
                               rescale = 1.0/255.0,
                               zoom_range=0.1,
                               shear_range = 0.1)

test_gen = ImageDataGenerator(rotation_range = 2,
                              horizontal_flip = True,
                              vertical_flip = False,
                              rescale = 1.0/255.0,
                              zoom_range=0.1,
                              shear_range = 0.1)
```
### One Hot Encoding Outputs
```
ytrain_onehot = utils.to_categorical(ytrain,10)
ytest_onehot = utils.to_categorical(ytest,10)
```
### Import VGG-19 model & add dense layers
```
base_model =  VGG19(include_top=False,
                    weights="imagenet",
                    input_tensor=None,
                    input_shape=(32,32,3),
                    pooling=None,)

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1024,activation=("relu")))
model.add(Dense(512,activation=("relu")))
model.add(Dense(256,activation=("relu")))
model.add(Dense(128,activation=("relu")))
model.add(Dense(64,activation=("relu")))
model.add(Dense(32,activation=("relu")))
model.add(Dense(10,activation=("relu")))
model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics="accuracy")

train_img_gen  = train_gen.flow(xtrain,ytrain_onehot,
                               batch_size = 64)		 
test_img_gen  = test_gen.flow(xtest,ytest_onehot,
                              batch_size = 64)

model.fit(train_img_gen,epochs = 25,
          validation_data = test_img_gen)
```
### Metrics
```
metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(test_img_gen), axis=1)

print(confusion_matrix(ytest,x_test_predictions))

print(classification_report(ytest,x_test_predictions))
```




## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
Include your plot here
</br>
</br>
</br>
### Classification Report
Include Classification Report here
</br>
</br>
</br>
### Confusion Matrix
Include confusion matrix here
</br>
</br>
</br>
## RESULT
</br>
</br>

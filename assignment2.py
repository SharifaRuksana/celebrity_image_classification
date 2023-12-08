import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

image_directory = r"C:\Users\shari\S3\Deep Learning\DL\Assignment_2\Dataset_Celebrities\cropped"
mess_dir = os.listdir(image_directory+ '/lionel_messi')
maria_dir= os.listdir(image_directory+ '/maria_sharapova')
roger_dir= os.listdir(image_directory+ '/roger_federer')
serena_dir= os.listdir(image_directory+ '/serena_williams')
virat_dir= os.listdir(image_directory+ '/virat_kohli')


print('The length of Messi image data is',len(mess_dir))
print('The length of Maria image data is',len(maria_dir))
print('The length of Roger image data is',len(roger_dir))
print('The length of Virat image data is',len(virat_dir))
print('The length of Serena image data is',len(serena_dir))

dataset = []
label = []
img_size=(224,224)

for i, image_name in tqdm(enumerate(mess_dir),desc = "Lionel Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_directory+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in tqdm(enumerate(maria_dir),desc = "Maria Sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_directory+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(1)

for i, image_name in tqdm(enumerate(roger_dir),desc = "Roger federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_directory+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(2)

for i, image_name in tqdm(enumerate(serena_dir),desc = "Serena Williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_directory+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(3)

for i, image_name in tqdm(enumerate(virat_dir),desc = "Virat Kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_directory+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(4)

dataset = np.array(dataset)
label = np.array(label)
print("dataset length", len(dataset))
print("label length", len(label))

print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)

print("Normalaising the Dataset. \n")

x_train=x_train.astype('float')/255
x_test=x_test.astype('float')/255 

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32,kernel_size =(3,3),activation="ReLU",input_shape=(224,224,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters=64,kernel_size =(3,3),activation="ReLU"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters=128,kernel_size =(3,3),activation="ReLU"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(filters=128,kernel_size =(3,3),activation="ReLU"))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512,activation="ReLU"))          
model.add(keras.layers.Dense(5,activation="softmax"))

model.summary()
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
#early_stop = tf.keras.callbacks.EarlyStopping(monitor="accuracy", mode="min",patience=10)
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=10,batch_size =64,validation_split=0.1)
print("Training Finished.\n")

def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((224,224))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction,axis=1)[0]
    class_name = ['lionel_messi','maria_sharapova','roger_federer','virat_kohli','serena_williams']
    predicted_class_name = class_name[predicted_class]
    return predicted_class_name

    



image_paths = [
    r'C:\Users\shari\S3\Deep Learning\DL\Assignment_2\Dataset_Celebrities\cropped\maria_sharapova\maria_sharapova6.png',
    r'C:\Users\shari\S3\Deep Learning\DL\Assignment_2\Dataset_Celebrities\cropped\lionel_messi\lionel_messi1.png',
    r'C:\Users\shari\S3\Deep Learning\DL\Assignment_2\Dataset_Celebrities\cropped\roger_federer\roger_federer8.png',
    r'C:\Users\shari\S3\Deep Learning\DL\Assignment_2\Dataset_Celebrities\cropped\serena_williams\serena_williams11.png',
    r'C:\Users\shari\S3\Deep Learning\DL\Assignment_2\Dataset_Celebrities\cropped\virat_kohli\virat_kohli9.png'
]

for image_path in image_paths:
    image_filename = os.path.basename(image_path)  # Extract filename from image path
    predicted_class_label = make_prediction(image_path, model)
    print(f"Predicted label for image {image_filename}: {predicted_class_label}")
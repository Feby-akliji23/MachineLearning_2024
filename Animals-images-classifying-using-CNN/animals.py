from google.colab import drive
drive.mount('/content/drive')

# membaca file
data_path ='/content/drive/MyDrive/Animals '

# Import libraries
import os  # Untuk berinteraksi dengan sistem file
import shutil  # Untuk mengelola file dan direktori secara lintas platform
import keras  # Untuk membangun model pembelajaran mendalam
import numpy as np  # Untuk operasi numerik pada array
from glob import glob  # Untuk menemukan file paths
from tqdm import tqdm  # Untuk progress bars

# Data preprocessing
from keras.preprocessing.image import ImageDataGenerator  # For image data augmentation

# Data visualization
import seaborn as sns  # Untuk statistical visualizations
import plotly.graph_objs as go  # Untuk interactive visualizations
import matplotlib.pyplot as plt  # Untuk membuat static plots
# Model architecture
from keras import Sequential  # Untuk building sequential models
from keras.models import load_model  # Untuk loading pre-trained models
from keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout  # Untuk defining model layers

# Training callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping  # Untuk training callbacks

# Pre-trained models
from tensorflow.keras.applications import InceptionV3, ResNet152V2  # Untuk using pre-trained models

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# List semua file di folder
folders = os.listdir(data_path)
# mendapatkan nomor di setiap kelas
class_sizes = []
for name in folders:
    class_size = len(os.listdir(data_path + "/" + name))
    class_sizes.append(class_size)

# cetak class distribution
print("Class Distribution:\n", class_sizes)

"""#from scratch

"""

from PIL import Image
import cv2
data=[]
labels=[]

def prepare_data(path, i):
  animals=os.listdir(path)
  for animal in animals:
    imag=cv2.imread(path+"/"+animal)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((100, 100))
    data.append(np.array(resized_image))
    labels.append(i)

prepare_data('/content/drive/MyDrive/Animals /horse', 0)
prepare_data("/content/drive/MyDrive/Animals /dog", 1)
prepare_data("/content/drive/MyDrive/Animals /butterfly", 2)
prepare_data("/content/drive/MyDrive/Animals /chicken", 3)
prepare_data("/content/drive/MyDrive/Animals /elephant", 4)
prepare_data("/content/drive/MyDrive/Animals /cat", 5)
prepare_data("/content/drive/MyDrive/Animals /spider", 6)
prepare_data("/content/drive/MyDrive/Animals /squirrel", 7)

animals=np.array(data)
labels=np.array(labels)

np.save("animals",animals)
np.save("labels",labels)
print(animals.shape)

s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]

num_classes=len(np.unique(labels))
data_length=len(animals)

(x_train,x_test)=animals[(int)(0.3*data_length):],animals[:(int)(0.3*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)
print(x_train.shape)

(y_train,y_test)=labels[(int)(0.3*data_length):],labels[:(int)(0.3*data_length)]
print(y_train.shape)

import keras
#One hot encoding
y_train_1d=y_train
y_test_1d=y_test
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

print(y_test_1d)

# import sequential model and all the required layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
#make model
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.7))
model.add(Dense(8,activation="softmax"))
model.summary()

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

from keras.callbacks import ModelCheckpoint, EarlyStopping
cbs = [
    EarlyStopping(patience=3, restore_best_weights=True,monitor="loss"),
    #ModelCheckpoint(name + ".h5", save_best_only=True)
]

saved_model=model.fit(x_train,y_train,batch_size=50,epochs=50,verbose=1, callbacks=cbs)

score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])
print('\n', 'Test F1_score:',score[2])
print('\n', 'Test percision:',score[3])
print('\n', 'Test recall:',score[4])

from sklearn.metrics import accuracy_score , confusion_matrix, classification_report

y_pred = (model.predict(x_test) > 0.5).astype(int)
y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7])
print(y_pred)
conf_matrix = confusion_matrix(y_test_1d, y_pred)

#Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Confusion Matrix:')
print(conf_matrix)

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

print(f'\nWeighted Precision: {precision:.4f}')
print(f'Weighted Recall: {recall:.4f}')
print(f'Weighted F1 Score: {f1:.4f}')

"""#pre_trained model"""

data_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    validation_split=0.2)

train_data = data_generator.flow_from_directory(
    data_path,
    target_size=(256,256),
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    subset='training')

# Muat data validasi dari direktori yang ditentukan dan terapkan generator
# subset: menentukan subset data yang akan dimuat, dalam hal ini, set validasi
valid_data = data_generator.flow_from_directory(
    data_path,
    target_size=(256,256),
    batch_size=32,
    shuffle=True,
    subset='validation')

# Tentukan nama model sebagai "Inception".
name = "Inception"

# Muat model InceptionV3 yang telah dilatih sebelumnya, freeze bobotnya, dan kecualikan lapisan klasifikasi terakhirnya.
base_model = InceptionV3(include_top=False, input_shape=(256,256,3), weights='imagenet')
base_model.trainable = False

# Buat model sekuensial dengan model dasar InceptionV3, lapisan pengumpulan rata-rata global, dua lapisan yang terhubung sepenuhnya, dan lapisan klasifikasi softmax akhir.
inception_model = Sequential([
    base_model,
    GAP(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='softmax')
], name=name)

# Kompilasi model dengan entropi silang kategoris jarang sebagai fungsi kerugian, pengoptimal Adam, dan akurasi sebagai metrik evaluasi.
inception_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc',precision_m, recall_m]
)

# Siapkan callback EarlyStopping dan ModelCheckpoint untuk memantau proses pelatihan dan menyimpan bobot model terbaik.
cbs = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(name + ".h5", save_best_only=True)
]

# Latih model menggunakan set data pelatihan dan validasi, menggunakan 50 epoch dan callback yang telah ditentukan sebelumnya.
inception_model.fit(
    train_data, validation_data=valid_data,
    epochs=5, callbacks=cbs
)

score = inception_model.evaluate(valid_data)
print('\n', 'Test accuracy:', score[1])
print('\n', 'Test F1_score:',score[2])
print('\n', 'Test percision:',score[3])
print('\n', 'Test recall:',score[4])
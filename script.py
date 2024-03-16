import tensorflow as tf
import os 

#protect memory 
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

#remove dodgy images 
import cv2 
import imghdr
data_dir = 'data'

image_exts = ['jpeg','jpg','bmp', 'png']
image_exts[0]
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)

        except Exception as e:
            print('Issue with image {}'.format(image_path))
#building dataset and doing the preprocessing 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
print (batch[1])

fig, ax= plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx]) 
#accessories  = 0
# Bags = 1

############ SCALING DATA ############
scaled = batch[0]/255
print(scaled.min()) 
#data transformation
data = data.map(lambda x, y:(x/255,y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
#print(batch)

############ SPLITTING DATA ############
data_len = len(data)
train_size = int(data_len*.7)
val_size = int(data_len*.2)+1
test_size = int (data_len*.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


############ DEEP MODEL ############
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(16,(3,3),1,activation = 'relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile('adam', loss= tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])
print(model.summary())

############  TRAIN MODEL ############
logdir = 'logs'
tensboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
hist = model.fit(train, epochs=20, validation_data= val, callbacks = [tensboard_callback])

############  PLOT PEROFRMANCE MODEL ############

fig = plt.figure()
plt.plot(hist.history['accuracy'], color = 'teal', label = 'accuracy')
plt.plot(hist.history['val_accuracy'], color = 'orange', label = 'val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['loss'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()


for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precesion: {pre.result().numpy()},Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

############  TEST MODEL ############

img = cv2.imread()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_))
plt.show

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
np.expand_dims(resize,0)
yhat = model.predict(np.expand_dims(resize/255,0))

############  SAVE MODEL ############
model.save(os.path.join('models','SlashModel.h5'))
new_model = load_model('SlashModel.h5')
new_model.predict(np.expand_dims(resize/255, 0))
---
title: TensorFlow-代码合集
tags: explain
author: zikang
---
这是一个TensorFlow的代码合集

# 四层神经网络
## four-layer-net 权重函数
```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient
import pickle


class FourLayerNet:

    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3,output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1,hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, hidden_size3)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size3, output_size)
        self.params['b4'] = np.zeros(output_size)

    def predict(self, x):
        W1,W2,W3,W4 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4']
        b1,b2,b3,b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = sigmoid(a3)
        a4 = np.dot(z3, W4) + b4
        y = softmax(a4)
        
        return y
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        grads['W4'] = numerical_gradient(loss_W, self.params['W4'])
        grads['b4'] = numerical_gradient(loss_W, self.params['b4'])
        
        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
            
    def gradient(self, x, t):
        #W1, W2 = self.params['W1'], self.params['W2']
        #b1, b2 = self.params['b1'], self.params['b2']
        W1, W2, W3,W4 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4']
        b1, b2, b3,b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = sigmoid(a3)
        a4 = np.dot(z3, W4) + b4
        y = softmax(a4)
        
        # backward
        dy = (y - t) / batch_num
        grads['W4'] = np.dot(z3.T, dy)
        grads['b4'] = np.sum(dy, axis=0)

        da3 = np.dot(dy, W4.T)
        dz3 = sigmoid_grad(a3) * da3
        grads['W3'] = np.dot(z2.T, dz3)
        grads['b3'] = np.sum(dz3, axis=0)

        da2 = np.dot(dz3, W3.T)
        dz2 = sigmoid_grad(a2) * da2
        grads['W2'] = np.dot(z1.T, dz2)
        grads['b2'] = np.sum(dz2, axis=0)
        
        da1 = np.dot(dz2, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
```

## 计算权重 绘制图形
```python
import sys, os  
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定  
import numpy as np  
import matplotlib.pyplot as plt  
from dataset.mnist import load_mnist  
from four_layer_net import FourLayerNet

# 读入数据  
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = FourLayerNet(input_size=784, hidden_size1=50, hidden_size2=100, hidden_size3=200, output_size=10)

iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100  
learning_rate = 0.1

train_loss_list = []  
train_acc_list = []  
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):  
    batch_mask = np.random.choice(train_size, batch_size)  
    x_batch = x_train[batch_mask]  
    t_batch = t_train[batch_mask]  
      
    # 计算梯度  
    # grad = network.numerical_gradient(x_batch, t_batch)  
    grad = network.gradient(x_batch, t_batch)  
      
    # 更新参数  
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):  
        network.params[key] -= learning_rate * grad[key]  
      
    loss = network.loss(x_batch, t_batch)  
    train_loss_list.append(loss)  
      
    if i % iter_per_epoch == 0:  
        train_acc = network.accuracy(x_train, t_train)  
        test_acc = network.accuracy(x_test, t_test)  
        train_acc_list.append(train_acc)  
        test_acc_list.append(test_acc)  
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 保存参数  
network.save_params("params1.pkl")  
print("Saved Network Parameters!")  
# 绘制图形  
markers = {'train': 'o', 'test': 's'}  
x = np.arange(len(train_acc_list))  
plt.plot(x, train_acc_list, label='train acc')  
plt.plot(x, test_acc_list, label='test acc', linestyle='--')  
plt.xlabel("epochs")  
plt.ylabel("accuracy")  
plt.ylim(0, 1.0)  
plt.legend(loc='lower right')  
plt.show()  
```

## 识别数字
```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image



def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def get_singledata(str):
    img = Image.open(str).convert('L')
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28))
    arr = []

    for i in range(28):
        for j in range(28):
            pixel = 1.0 - float(img.getpixel((j, i))) / 255.0

            arr.append(pixel)
    arr1 = np.array(arr).reshape((784,))
    return arr1

def init_network():
    with open("params1.pkl", 'rb') as f:
    # with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3, W4 = network['W1'], network['W2'], network['W3'],network['W4']
    b1, b2, b3, b4 = network['b1'], network['b2'], network['b3'],network['b4']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = sigmoid(a3)
    a4 = np.dot(z3, W4) + b4
    y = softmax(a4)

    return y

x=get_singledata('2-44.jpg')
print("x=",x*255)
network = init_network()
y = predict(network, x)
print("y=",y)
p = np.argmax(y)  # 获取概率最高的元素的索引
print("p=",p)
```

# 模型的保存和加载

## 利用均方误差进行模型评价(直接加入代码结尾)
```python
y_test = pd.DataFrame(y_test)
pre = pd.DataFrame(pre)
mse = (sum(y_test - pre) ** 2) / 10
print('均方误差为：', mse)
```

## 线性模型-保存完整模型
```python
import tensorflow as tf
import pandas as pd

# 读取数据
data = pd.read_csv('../data/line_fit_data.csv').values
# 划分训练集和测试集
x = data[:-10, 0]
y = data[:-10, 1]
x_test = data[-10:, 0]
y_test = data[-10:, 1]

# 构建Sequential网络
model_net = tf.keras.models.Sequential()  # 实例化网络
model_net.add(tf.keras.layers.Dense(1, input_shape=(1, )))  # 添加全连接层
print(model_net.summary())

# 构建损失函数
model_net.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))

# 模型训练
model_net.fit(x, y, verbose=1, epochs=20, validation_split=0.2)
#print(x_test[0:1])
# HDF5格式文件保存整个模型
model_net.save('model_mnist.h5')
pre = model_net.predict(x_test[0:1])
#print(x_test[0:1])
print(pre)

# # 利用均方误差进行模型评价
# y_test = pd.DataFrame(y_test)
# pre = pd.DataFrame(pre)
# mse = (sum(y_test - pre) ** 2) / 10
# print('均方误差为：', mse)
```

## 线性模型-保存模型参数
```python
import tensorflow as tf
import pandas as pd

# 读取数据
data = pd.read_csv('../data/line_fit_data.csv').values
# 划分训练集和测试集
x = data[:-10, 0]
y = data[:-10, 1]
x_test = data[-10:, 0]
y_test = data[-10:, 1]

# 构建Sequential网络
model_net = tf.keras.models.Sequential()  # 实例化网络
model_net.add(tf.keras.layers.Dense(1, input_shape=(1, )))  # 添加全连接层
print(model_net.summary())

# 构建损失函数
model_net.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))

# 模型训练
model_net.fit(x, y, verbose=1, epochs=20, validation_split=0.2)
#print(x_test[0:1])
#只保存模型参数
model_net.save_weights('mannul_checkpoint')
pre = model_net.predict(x_test[0:1])

print(pre)
```

## 线性模型-加载整个模型
```python
import tensorflow as tf
import pandas as pd

# 读取数据
data = pd.read_csv('../data/line_fit_data.csv').values
# 划分训练集和测试集
x = data[:-10, 0]
y = data[:-10, 1]
x_test = data[-10:, 0]
y_test = data[-10:, 1]

#加载整个模型
model_net = tf.keras.models.load_model('model_mnist.h5')
pre = model_net.predict(x_test[0:1])
#print(x_test[0:1])
print(pre)
```

## 线性模型-加载模型参数
```python
import tensorflow as tf
import pandas as pd

# 读取数据
data = pd.read_csv('../data/line_fit_data.csv').values
# 划分训练集和测试集
x = data[:-10, 0]
y = data[:-10, 1]
x_test = data[-10:, 0]
y_test = data[-10:, 1]

# 构建Sequential网络
model_net = tf.keras.models.Sequential()  # 实例化网络
model_net.add(tf.keras.layers.Dense(1, input_shape=(1,)))  # 添加全连接层
# print(model_net.summary())
#
# # 构建损失函数
model_net.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))
# #
# # # 模型训练
# # model_net.fit(x, y, verbose=1, epochs=20, validation_split=0.2)

# 加载整个模型参数
# model_net = tf.keras.models.load_model('model_mnist.h5')
model_net.load_weights('mannul_checkpoint')
pre = model_net.predict(x_test[0:1])
# print(x_test[0:1])
print(pre)
```

# 四个模型

## LeNet-5
```python
#%%
#LeNet-5
#代码示例：5-1
# from keras.datasets import mnist
import tensorflow.compat.v1 as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist
(X0,Y0),(X1,Y1) = mnist.load_data()
print(X0.shape)
from matplotlib import pyplot as plt
plt.figure()
fig,ax = plt.subplots(2,5)
ax=ax.flatten()
for i in range(10):
    Im=X0[Y0==i][0]
    ax[i].imshow(Im)
plt.show();
```

```python
#%%
#代码示例：5-2
#from keras.utils import np_utils
import tensorflow.keras.utils as np_utils
N0=X0.shape[0];N1=X1.shape[0]
print([N0,N1])
X0 = X0.reshape(N0,28,28,1)/255
X1 = X1.reshape(N1,28,28,1)/255
YY0 = np_utils.to_categorical(Y0)
YY1 = np_utils.to_categorical(Y1)
print(YY1)
```
```python
#%%
#代码示例：5-3
#from keras.layers import Conv2D,Dense,Flatten,Input,MaxPooling2D
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Input,MaxPooling2D
from tensorflow.keras import Model

input_layer = Input([28,28,1])
x = input_layer
x = Conv2D(6,[5,5],padding = "same", activation = 'relu')(x)
x = MaxPooling2D(pool_size = [2,2], strides = [2,2])(x)
x = Conv2D(16,[5,5],padding = "valid", activation = 'relu')(x)
x = MaxPooling2D(pool_size = [2,2], strides = [2,2])(x)
x = Flatten()(x)
x = Dense(120,activation = 'relu')(x)
x = Dense(84,activation = 'relu')(x)
x = Dense(10,activation = 'softmax')(x)
output_layer=x
model=Model(input_layer,output_layer)
model.summary()
```
```python
#%%
#代码示例：5-4
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X0, YY0, epochs=10, batch_size=200, validation_data=[X1,YY1])
```

## AlexNet
```python
#%%
#AlexNet
#代码示例：5-5
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMSIZE=227

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'ChineseStyle/test/',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')

train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'ChineseStyle/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=200,
    class_mode='categorical')
```
```python
#%%
#代码示例：5-6
from matplotlib import pyplot as plt

plt.figure()
fig,ax = plt.subplots(2,5)
fig.set_figheight(7)
fig.set_figwidth(15)
ax=ax.flatten()
X,Y=next(validation_generator)
for i in range(10): ax[i].imshow(X[i,:,:,:])
```
```python
#%%
#代码示例：5-7
from tensorflow.keras.layers import Activation,Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
from tensorflow.keras import Model

IMSIZE = 227
input_layer = Input([IMSIZE,IMSIZE,3])
x = input_layer
x = Conv2D(96,[11,11],strides = [4,4], activation = 'relu')(x)
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Conv2D(256,[5,5],padding = "same", activation = 'relu')(x)
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x)
x = Conv2D(384,[3,3],padding = "same", activation = 'relu')(x)
x = Conv2D(256,[3,3],padding = "same", activation = 'relu')(x)
x = MaxPooling2D([3,3], strides = [2,2])(x)
x = Flatten()(x)
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(2,activation = 'softmax')(x)
output_layer=x
model=Model(input_layer,output_layer)
model.summary()
```
```python
#%%
#代码示例：5-8
from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.fit_generator(train_generator,epochs=5,validation_data=validation_generator)
```

## VGG
```python
#%%
# 代码示例：5-9
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMSIZE = 224

train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
    'data_vgg/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size=32,
    class_mode='categorical')

validation_generator = ImageDataGenerator(
    rescale=1. / 255).flow_from_directory(
        'data_vgg/test',
        target_size=(IMSIZE, IMSIZE),
        batch_size=32,
        class_mode='categorical')
```
```python
#%%
#代码示例：5-10
from matplotlib import pyplot as plt

plt.figure()
fig, ax = plt.subplots(2, 5)
fig.set_figheight(6)
fig.set_figwidth(15)
ax = ax.flatten()
X, Y = next(validation_generator)
for i in range(10):
    ax[i].imshow(X[i, :, :, ])
```
```python
#%%
#代码示例：5-11(书稿的印刷中疏忽了全连接层)
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

IMSIZE = 224
input_shape = (IMSIZE, IMSIZE, 3)
input_layer = Input(input_shape)
x = input_layer

x = Conv2D(64, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(64, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(128, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(256, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = Conv2D(512, [3, 3], padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(4096, activation = "relu")(x)
x = Dense(4096, activation = "relu")(x)
x = Dense(200, activation = "softmax")(x)
output_layer = x
model_vgg16 = Model(input_layer, output_layer)
model_vgg16.summary()
```
```python
#%%
#代码示例：5-12
from tensorflow.keras.optimizers import Adam
model_vgg16.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
#model_vgg16.fit_generator(train_generator,epochs=5,validation_data=validation_generator)
model_vgg16.fit(train_generator,epochs=5,validation_data=validation_generator)
```

## ResNet
### resnet_18_34.py
```python
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import tensorflow as tf

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=2):
        super(ResNet, self).__init__()
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)
    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

def resnet18(classes):
    net=ResNet([2, 2, 2, 2],num_classes=classes)
    return net

def resnet34(classes):
    net=ResNet([3, 4, 6, 3],num_classes=classes)
    return net

if __name__ == '__main__':
    print('this is a resnet_18_34')
```
### resnet_50_101_152.py
```python
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import tensorflow as tf

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (1, 1), strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.conv3 = layers.Conv2D(filter_num*4, (1, 1), strides=1, padding='same')
        self.bn3 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num*4, (1, 1), strides=1))
        else:
            self.downsample = lambda x: x
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=2):
        super(ResNet, self).__init__()
        self.stem = Sequential([layers.Conv2D(64, (7, 7), strides=(2, 2)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')
                                ])
        self.layer1 = self.build_resblock(64, layer_dims[0], stride=2)
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2) #这里的stride是为了identity是否需要变换维度，1为不需要变换，2需要
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)
    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

def resnet50(classes):
    net = ResNet([3, 4, 6, 3],num_classes=classes)
    return net

def resnet101(classes):
    net = ResNet([3, 4, 23, 3],num_classes=classes)
    return net

def resnet152(classes):
    net = ResNet([3, 8, 36, 3],num_classes=classes)
    return net

if __name__ == '__main__':
    print('this is a resnet_50_101_152')
```
### resnet_model.py
```python
import resnet_18_34,resnet_50_101_152
def resnet18(classes=2):
    return resnet_18_34.resnet18(classes)
def resnet34(classes=2):
    return resnet_18_34.resnet34(classes)
def resnet50(classes=2):
    return resnet_50_101_152.resnet50(classes)
def resnet101(classes=2):
    return resnet_50_101_152.resnet101(classes)
def resnet152(classes=2):
    return resnet_50_101_152.resnet152(classes)
if __name__ == '__main__':
    print("this is a model!")
```
### resnet_train.py
```python
import cv2,os
import numpy as np
import tensorflow as tf
from random import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers,losses,metrics
from resnet_model import resnet18,resnet34,resnet50,resnet101,resnet152

labels_num=3 # 类别数

#测试集的导入
def load_image(path,shape):
    img_list = []
    label_list = []
    label_name_dict={}
    dir_counter = 0
    # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        label_name_dict[child_path]=dir_counter
        for dir_image in os.listdir(child_path):
            img = cv2.imread(os.path.join(child_path, dir_image))
            img = img / 255.0
            img=cv2.resize(img,(shape[0],shape[1]))
            img_list.append(img)
            label_list.append(dir_counter)
        dir_counter += 1

    length= len(img_list)
    index = [i for i in range(length)]
    shuffle(index)  # 打乱索引
    img_np=np.array(img_list)
    label_np=np.array(label_list)
    img_np1 = img_np[index]
    label_np1 = label_np[index]
    train_l=int(0.7*length)

    train_data = np.array(img_np1)[0:train_l]
    train_label =np.array(label_np1)[0:train_l]
    test_data = np.array(img_np1)[train_l:length]
    test_label = np.array(label_np1)[train_l:length]
    return train_data,train_label,test_data,test_label,label_name_dict
def load_image_single(path,shape):
    img_list=[]
    img = cv2.imread(path)
    img = img / 255.0
    img = cv2.resize(img, (shape[0], shape[1]))
    img_list.append(img)
    print("img_list=",img_list)
    img_np = np.array(img_list)
    print("img_np=", img_np)
    return img_np
def model(label_num=labels_num):
    #网络层的搭建
    networks=resnet18(classes=label_num)
    return networks
def train(net,train_data,train_label):
    def get_batch(batch_size, i):
        x = batch_size * i
        train_data_batch = train_data[x:x + batch_size, :]
        train_lable_batch = train_label[x:x + batch_size]
        return train_data_batch, train_lable_batch

    #epoch = 5  # 迭代次数
    epoch = 5  # 迭代次数
    batch_size = 32  # 一批要处理的图像
    shape_t=train_data.shape
    net.build(input_shape=(batch_size,shape_t[1],shape_t[2],shape_t[3]))
    num_train_data = shape_t[0]  # 训练图像总数
    batch_num = int(num_train_data // batch_size)  # 训练批数：这里必须取整
    optimizer = optimizers.Adam(learning_rate=0.001)  # 该函数可以设置一个随训练进行逐渐减小的学习率，此处我们简单的设学习率为常量
    for n in range(epoch):
        for i in range(batch_num):
            with tf.GradientTape() as tape:  # with语句内引出需要求导的量
                x, y = get_batch(batch_size, i)
                out = net(x)
                y_onehot = tf.one_hot(y, depth=labels_num)  # 一维表示类别（0-9）-> 二维表示类别（1,0,0,0，...）...
                loss_object = losses.CategoricalCrossentropy(from_logits=True)  # 交叉熵损失函数.这是一个类，loss_object为类的实例化对象
                loss = loss_object(y_onehot, out)  # 使用损失函数类来计算损失
                print('epoch:%d batch:%d loss:%f' % (n, i, loss.numpy()))
            grad = tape.gradient(loss, net.trainable_variables)  # 用以自动计算梯度. loss对网络中的所有参数计算梯度
            optimizer.apply_gradients(zip(grad, net.trainable_variables))  # 根据梯度更新网络参数
    #net.save('model/resnet.h5')
    net.save_weights('model/model_weight')

def test(test_data,test_label):
    #net=load_model('model/resnet.h5')
    net = model()
    net.load_weights('model/model_weight')
    batch_size=32
    s_c_a = metrics.SparseCategoricalAccuracy()  # metrics用于监测性能指标，这里用update_state来对比
    num_test_batch = int(test_data.shape[0] // batch_size)  # 测试集数量
    for num_index in range(num_test_batch):
        start_index, end_index = num_index * batch_size, (num_index + 1) * batch_size  # 每一批的起始索引和结束索引
        y_predict = net.predict(test_data[start_index:end_index])
        s_c_a.update_state(y_true=test_label[start_index:end_index], y_pred=y_predict)
    print('test accuracy:%f' % s_c_a.result())
def predict(img_data):
    net = model()
    #net.load_weights('model_weight')
    net.load_weights('model/model_weight')
    batch_size = 1
    y_predict = net.predict(img_data)
    return y_predict
if __name__ == '__main__':
    #训练及评测
    path = "validation"
    # #train_data,train_label,test_data,test_label=load_image(path,(244,244))
    # #train_data, train_label, test_data, test_label = load_image(path, (128, 128))
    # #train_data, train_label, test_data, test_label, label_name_dict= load_image(path, (128, 128))
    train_data, train_label, test_data, test_label, label_name_dict = load_image(path, (244, 244))
    # print("label_name_dict=",label_name_dict)
    # net = model()
    # train(net,train_data,train_label)
    # print('------------------------------')
    # test(test_data,test_label)

    #单个测试
    img_data=load_image_single("2960610406_b61930727f_n.jpg", (244, 244))
    print(predict(img_data))
    result=np.argmax(predict(img_data))
    print(result)
    print(label_name_dict)
```

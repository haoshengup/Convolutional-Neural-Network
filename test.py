import cnn
import numpy as np
import re

training_data = np.loadtxt(open('training_data.csv', 'r'), delimiter=',', skiprows=0)
target = np.loadtxt(open('target.csv', 'r'), delimiter=',', skiprows=0)
data = training_data.reshape(1000,28,28)[:,np.newaxis,:,:]
t = np.zeros(shape=(10,1000))
train_data = data[:900,:,:]
test_data = data[900:1000,:,:]
train_t = t[:,:900]
test_t = t[:,900:1000]
for i in range(len(target)):
    if target[i] == 0:
        t[:,i][0] = 1
    elif target[i] == 1:
        t[:,i][1] = 1
    elif target[i] == 2:
        t[:, i][2] = 1
    elif target[i] == 3:
        t[:, i][3] = 1
    elif target[i] == 4:
        t[:,i][4] = 1
    elif target[i] == 5:
        t[:, i][5] = 1
    elif target[i] == 6:
        t[:, i][6] = 1
    elif target[i] == 7:
        t[:,i][7] = 1
    elif target[i] == 8:
        t[:, i][8] = 1
    elif target[i] == 9:
        t[:, i][9] = 1

net = cnn.NetWork([cnn.ConvolutionLayer(filter_shape=[1,5,5],conv_depth=6,image_shape=[1,28,28]), cnn.PoolLayer(), cnn.FullyConnectedLayer([864,100,10])])
e_mv_out = net.SGD(train_data, train_t, 0.02, 400, 50)
net.evaluate(train_data, train_t, e_mv_out)
net.evaluate(test_data, test_t, e_mv_out)

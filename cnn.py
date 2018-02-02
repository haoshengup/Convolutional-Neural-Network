#!/usr/bin/python
#description : A simple Python script of neural network
#author : haosheng
#date : 20180202
#version : 0.1
#usage : NetWork_Adam.py
#python_version : 3.6.3

import numpy as np
from scipy import signal
import re
import random

class NetWork(object):
    def __init__(self,layers_list):
        '''
        :param layers_list: initialize every layer of network
        '''
        self.layers_list = layers_list

    def forward(self, inpt_array, mode = 'train', e_mv_out = None):
        
        out = [inpt_array]
        if mode == 'train':
            for layers in self.layers_list:
                if isinstance(inpt_array, np.ndarray):
                    inpt_array = layers.forward(inpt_array)
                else:
                    inpt_array = layers.forward(inpt_array[0])
                out.append(inpt_array)
            return out
        else:
            layers_list_name = [str(name) for name in self.layers_list]
            e_mv_out_copy = e_mv_out[:]
            for k, layers in enumerate(self.layers_list):
                if re.match(r'.*ConvolutionLayer', layers_list_name[k]):
                    inpt_array = layers.forward(inpt_array, mode='test',e_mv_out=e_mv_out_copy[0])
                    del e_mv_out_copy[0]
                elif re.match(r'.*PoolLayer', layers_list_name[k]):
                    inpt_array = layers.forward(inpt_array)
                else:
                    inpt_array = layers.forward(inpt_array, mode='test', e_mv_out=e_mv_out_copy[0])
                out.append(inpt_array)
            return out


    def backward(self, out, target, eta):
       
        layers_list_reverse = self.layers_list[::-1]
        layers_list_reverse_name = [str(name) for name in layers_list_reverse]
        out_reverse = out[::-1]
        for j, layers in enumerate(layers_list_reverse):
            if re.match(r'.*FullyConnectedLayer',layers_list_reverse_name[j]):
                back_delta = layers.backward(out_reverse[1], out_reverse[0],target, eta)
            elif re.match(r'.*PoolLayer', layers_list_reverse_name[j]):
                back_delta = layers.backward(back_delta)
            else:
                back_delta = layers.backward(out_reverse[j], out_reverse[j + 1], back_delta, eta)

    def SGD(self, training_data, target, eta, mini_batch_size, epoches = 300):
        n = training_data.shape[0]
        cost_total_list =[]
        out_list = []
        all_conv_mv_list = []
        full_mv_list = []
        e_mv_out = []

        index = [i for i in range(n)]
        random.shuffle(index)
        training_data = training_data[index]
        target = target[:, index]
        mini_batches = [(training_data[k:k + mini_batch_size], target[:, k:k + mini_batch_size]) for k in range(0, n, mini_batch_size)]
        for j in range(epoches):
            cost_total = 0
            if j < epoches - 1:
                for mini_batch in mini_batches:
                    out = self.forward(mini_batch[0])
                    self.backward(out, mini_batch[1], eta)
                    cost_total = cost_total + self.layers_list[-1].cost(out[-1][0][-1], mini_batch[1])
            else:
                for mini_batch in mini_batches:
                    out = self.forward(mini_batch[0])
                    self.backward(out, mini_batch[1], eta)
                    cost_total = cost_total + self.layers_list[-1].cost(out[-1][0][-1], mini_batch[1])
                    out_list.append(out)
            print('Epoch {0}: cost = {1}'.format(j, cost_total/n))
            cost_total_list.append(cost_total/n)
        layers_list_name = [str(name) for name in self.layers_list]
        for k in range(len(self.layers_list)):
            if re.match(r'.*ConvolutionLayer', layers_list_name[k]):
                conv_mv_list = [ol[k + 1][3] for ol in out_list]
                all_conv_mv_list.append(conv_mv_list)
            elif re.match(r'.*FullyConnectedLayer', layers_list_name[k]):
                full_mv_list = [ol[k + 1][3] for ol in out_list]

        for k in range(len(all_conv_mv_list)):
            conv_mv_array = np.array(all_conv_mv_list[k])
            conv_mv_out = conv_mv_array.mean(axis=0)
            conv_mv_out[1,:] = conv_mv_out[1,:] * mini_batch_size / (mini_batch_size - 1)
            e_mv_out.append(conv_mv_out)

        full_mv_array = np.array(full_mv_list)
        full_mv_out = full_mv_array.mean(axis=0)
        full_mv_out = full_mv_out.tolist()
        for k in range(len(full_mv_out)):
            full_mv_out[k][:,1] = full_mv_out[k][:,1] * mini_batch_size / (mini_batch_size - 1)
        e_mv_out.append(full_mv_out)
        return e_mv_out


    def evaluate(self, test_data, test_target, e_mv_out):
        '''
        evaluate the test_data
        :param test_data: test data
        :param test_target: target
        :return: nums to identified correctly, nums to identified wrongly, recogniton rate
        '''
        out = self.forward(test_data, mode='test', e_mv_out=e_mv_out)
        out_max_index = np.argmax(out[-1][-1], axis=0)
        total_num = test_target.shape[1]
        correct_num = np.sum((out_max_index == np.argmax(test_target, axis=0)) == 1)
        print('correct num：{0}\n'.format(correct_num))
        print('wrong num：{0}\n' .format(total_num - correct_num))
        print('recognition rate：{0}\n' .format(correct_num/total_num))

class PoolLayer(object):
    def __init__(self, mode = 'max_pooling', size = 2):
        '''
        initialize poollayer
        :param mode: mode of subsumpling:'max_pooling' or 'mean_pooling'
        :param size: size of pool
        '''
        self.mode = mode
        self.size = size

    def forward(self, inpt_array):
       
        inpt_size = inpt_array[0].shape
        out_array = np.zeros(shape=(inpt_array.shape[0], inpt_size[0], int(inpt_size[1]/self.size), int(inpt_size[2]/self.size)))
        poollayer_depth = inpt_size[0]
        pool_data = np.zeros(shape = (poollayer_depth, int(inpt_size[1]/self.size), int(inpt_size[2]/self.size)))
        row_num = list(range(0, inpt_size[1] - 1, self.size))
        line_num = list(range(0, inpt_size[2] - 1, self.size))
        for index in range(inpt_array.shape[0]):
            for layer in range(poollayer_depth):
                row = 0
                line = 0
                for ln in row_num:
                    for rn in line_num:
                        if self.mode == 'max_pooling':
                            pool_data[layer, row, line] = np.max(inpt_array[index][layer, ln : ln + self.size, rn : rn + self.size])
                        else:
                            pool_data[layer, row, line] = np.mean(inpt_array[index][layer, ln: ln + self.size, rn: rn + self.size])
                        line = line + 1
                    line = 0
                    row = row + 1
            out_array[index] = pool_data
        return out_array

    def backward(self, inpt):
       
        pool_array = np.ones(shape = (self.size, self.size))
        if self.mode == 'max_pooling':
            back_delta = np.kron(inpt, pool_array)
        else:
            back_delta = np.kron(inpt/(self.size ** 2), pool_array)
        return back_delta

class ConvolutionLayer(object):
    def __init__(self, filter_shape, conv_depth, image_shape, stride = 1):
        '''
        :param filter_shape: the size of the filter kernel, a list and the lenth is 3
        :param conv_depth: the depth of filter kernel
        :param image_shape: the size of input data, a list and the lenth is 3 
        :param stride: the stride of the filter kernel
        '''
        self.filer_shape = filter_shape
        self.conv_depth = conv_depth
        self.image_shape = image_shape
        self.stride = stride
        self.weights = np.random.randn(conv_depth, filter_shape[0], filter_shape[1], filter_shape[2])
        self.gamma = np.ones(conv_depth)
        self.beta = np.zeros(conv_depth)
        self.epsilon = 0.001

    def forward(self, inpt_array, mode = 'train', e_mv_out = 'None'):
       
        out_array = np.zeros(shape=(inpt_array.shape[0], self.conv_depth, int((inpt_array.shape[2] - self.filer_shape[1])/self.stride) + 1, int((inpt_array.shape[3] - self.filer_shape[2])/self.stride) + 1))
        net = np.zeros_like(out_array)
        x_norm = np.zeros_like(out_array)
        y = np.zeros_like(out_array)

        if mode == 'train':
            for index in range(inpt_array.shape[0]):               
                conv_list = [self.convolve2d_depth(inpt_array[index], self.weights[d, :, :, :], func='forward') for d in range(self.conv_depth)]
                net[index] = np.array(conv_list)
            net_mean = [net[:,d,:,:].mean() for d in range(self.conv_depth)]
            net_std = [net[:,d,:,:].std() for d in range(self.conv_depth)]
            net_var = list(map(lambda x: x**2, net_std))

            for d in range(self.conv_depth):
                x_norm[:,d,:,:] = (net[:,d,:,:] - net_mean[d]) / np.sqrt(net_var[d] + self.epsilon)
                y[:,d,:,:] = self.gamma[d] * x_norm[:,d,:,:] + self.beta[d]
            out_array = self.activation_function(y)

            return out_array, net, x_norm, [net_mean, net_var]
        else:
            for index in range(inpt_array.shape[0]):                
                conv_list = [self.convolve2d_depth(inpt_array[index], self.weights[d, :, :, :], func='forward') for d in
                             range(self.conv_depth)]
                net[index] = np.array(conv_list)

            for d in range(self.conv_depth):
                x_norm[:, d, :, :] = (net[:, d, :, :] - e_mv_out[0,d]) / np.sqrt(e_mv_out[1,d] + self.epsilon)
                y[:, d, :, :] = self.gamma[d] * x_norm[:, d, :, :] + self.beta[d]
            out_array = self.activation_function(y)

            return out_array

    def backward(self, out_forward, inpt_array, inpt_delta, eta):
       
        lambda_weights_array = np.zeros(shape=(inpt_array.shape[0], self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]))
        back_delta_aray = np.zeros(shape=inpt_array.shape)

        y_derivative = inpt_delta
        x_norm_derivative = np.zeros_like(y_derivative)
        net_derivative = np.zeros_like(y_derivative)
        var_derivative = inpt_delta.shape[1] * [0]
        mean_derivative = inpt_delta.shape[1] * [0]
        gamma_derivative = np.zeros(inpt_delta.shape[1])
        beta_derivative = np.zeros(inpt_delta.shape[1])
        m = inpt_delta.shape[0] * inpt_delta.shape[2] * inpt_delta.shape[3]

        for d in range(inpt_delta.shape[1]):
            x_norm_derivative[:,d,:,:] = y_derivative[:,d,:,:] * self.gamma[d]
            var_derivative[d] = (x_norm_derivative[:,d,:,:] * (out_forward[1][:,d,:,:] - out_forward[3][0][d]) * (-0.5) * pow(out_forward[3][1][d] + self.epsilon, -1.5)).sum()
            mean_derivative[d] = (x_norm_derivative[:,d,:,:] * (-1) / np.sqrt(out_forward[3][1][d] + self.epsilon)).sum() + \
                                    var_derivative[d] * ((-2) * (out_forward[1][:,d,:,:] - out_forward[3][0][d])).sum() / m
            net_derivative[:,d,:,:] = x_norm_derivative[:,d,:,:] / np.sqrt(out_forward[3][1][d] + self.epsilon) + \
                                var_derivative[d] * 2 * (out_forward[1][:,d,:,:] - out_forward[3][0][d])/m + mean_derivative[d]/m
            gamma_derivative[d] = (y_derivative[:,d,:,:] * out_forward[2][:,d,:,:]).sum()
            beta_derivative[d] = y_derivative[:,d,:,:].sum()
        inpt_delta = net_derivative

        for col in range(inpt_array.shape[0]):
            lambda_weights = [self.rotate180(self.convolve2d_depth(inpt_array[col], self.rotate180(inpt_delta[col,d,:,:]), func='backward_gradient')) for d in range(self.conv_depth)]
            lambda_weights = np.array(lambda_weights)
            lambda_weights_array[col] = lambda_weights
           
            back_delta = [self.convolve2d_depth(inpt_delta[col,d,:,:], self.rotate180(self.weights[d,:,:,:]), func='backward_delta',mode='full') * self.activation_func_derivative(inpt_array[col]) for d in range(self.conv_depth)]
            back_delta = sum(back_delta)
            back_delta_aray[col] = back_delta
       
        self.weights = self.weights - eta * sum(lambda_weights_array)/inpt_array.shape[0]
        self.gamma = self.gamma - eta * gamma_derivative
        self.beta = self.beta - eta * beta_derivative
        return back_delta_aray

    def convolve2d_depth(self, inpt, conv_kernal, func, mode = 'valid'):
       
        if func == 'forward':
            conv_list = [signal.convolve2d(inpt[d, :, :], conv_kernal[d, :, :], mode) for d in range(inpt.shape[0])]
            return  sum(conv_list)
        elif func == 'backward_gradient':
            conv_list = [signal.convolve2d(inpt[d, :, :], conv_kernal, mode) for d in range(inpt.shape[0])]
            return np.array(conv_list)
        else:
            conv_list = [signal.convolve2d(inpt, conv_kernal[d, :, :], mode) for d in range(conv_kernal.shape[0])]
            return np.array(conv_list)

    def activation_function(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def activation_func_derivative(self, out):
        return out * (1 - out)

    def rotate180(self, matrix):
        left_right = list(map(lambda x: x[::-1], matrix)) 
        up_down = left_right[::-1] 
        return np.array(up_down)

class FullyConnectedLayer(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(next_layer_num, previous_layer_num) / np.sqrt(previous_layer_num) \
                        for previous_layer_num, next_layer_num in zip(sizes[:-1], sizes[1:])]
        self.gamma = [np.ones((node, 1)) for node in self.sizes]
        self.beta = [np.zeros((node, 1)) for node in self.sizes]
        self.epsilon = 0.001

    def forward(self, inpt_array, mode = 'train', e_mv_out = None):
        inpt_num = inpt_array[0].size 
        data_num = inpt_array.shape[0] 
        a = np.zeros(shape=(inpt_num, data_num))
        for col in range(data_num):
            a[:,col] = inpt_array[col].reshape((inpt_num, 1))[:,0]

        out = [a]
        net_list = [a]
        mv_list = [np.zeros(shape=(a.shape[0], 2))]
        x_norm_list = [a]

        if mode == 'train':
            for w, gamma, beta in zip(self.weights[:-1], self.gamma[1:-1], self.beta[1:-1]):
                net = np.dot(w, a)
                net_list.append(net)
                net_mean = net.mean(axis=1)[:, np.newaxis]
                net_std = net.std(axis=1)[:, np.newaxis]
                net_var = pow(net_std, 2)
                x_norm = (net - net_mean) / np.sqrt(net_var + self.epsilon)
                x_norm_list.append(x_norm)
                mv_list.append(np.hstack((net_mean, net_var)))
                y = gamma * x_norm + beta
                a = self.relu(y)
                out.append(a)
            net = np.dot(self.weights[-1], out[-1])
            net_list.append(net)
            net_mean = net.mean(axis=1)[:, np.newaxis]
            net_std = net.std(axis=1)[:, np.newaxis]
            net_var = pow(net_std, 2)
            x_norm = (net - net_mean) / np.sqrt(net_var + self.epsilon)
            x_norm_list.append(x_norm)
            mv_list.append(np.hstack((net_mean, net_var)))
            y = self.gamma[-1] * x_norm + self.beta[-1]
            out.append(self.softmax(y))
            return out, net_list, x_norm_list, mv_list
        else:
            for w, gamma, beta, e_mv in zip(self.weights[:-1], self.gamma[1:-1], self.beta[1:-1], e_mv_out[1:-1]):
                net = np.dot(w, a)
                x_norm = (net - e_mv[:, 0][:, np.newaxis]) / np.sqrt(e_mv[:, 1][:, np.newaxis] + self.epsilon)
                y = gamma * x_norm + beta
                a = self.relu(y)
                out.append(a)
            net = np.dot(self.weights[-1], out[-1])
            x_norm = (net - e_mv_out[-1][:, 0][:, np.newaxis]) / np.sqrt(e_mv_out[-1][:, 1][:, np.newaxis] + self.epsilon)
            y = self.gamma[-1] * x_norm + self.beta[-1]
            out.append(self.softmax(y))
            return out

    def backward(self, inpt_array, out_forward, target, eta):
       
        delta = [np.zeros((node, target.shape[1])) for node in self.sizes]
        y_derivative = [np.zeros((node, target.shape[1])) for node in self.sizes]
        x_norm_derivative = [np.zeros((node, target.shape[1])) for node in self.sizes]
        var_derivative = [np.zeros((node, 1)) for node in self.sizes]
        mean_derivative = [np.zeros((node, 1)) for node in self.sizes]
        net_derivative = [np.zeros((node, target.shape[1])) for node in self.sizes]

        y_derivative[-1] = (self.cost_derivative(out_forward[0][-1], target)) * self.softmax_derivative(out_forward[0][-1])
        x_norm_derivative[-1] = y_derivative[-1] * self.gamma[-1]
        var_derivative[-1] = (x_norm_derivative[-1] * (out_forward[1][-1] - out_forward[3][-1][:, 0][:, np.newaxis]) *  \
                              (-0.5) * pow(out_forward[3][-1][:, 1][:, np.newaxis] + self.epsilon, -1.5)).sum(axis=1)[:, np.newaxis]
        mean_derivative[-1] = (x_norm_derivative[-1] * (-1) / np.sqrt(out_forward[3][-1][:, 1][:, np.newaxis] + self.epsilon)).sum(axis=1)[:, np.newaxis] \
                              + var_derivative[-1] * (-2 * (out_forward[1][-1] - out_forward[3][-1][:, 0][:, np.newaxis])).sum(axis=1)[:, np.newaxis] / target.shape[1]
        net_derivative[-1] = x_norm_derivative[-1] / np.sqrt(out_forward[3][-1][:, 1][:, np.newaxis] + self.epsilon) + \
                             var_derivative[-1] * 2 * (out_forward[1][-1] - out_forward[3][-1][:, 0][:, np.newaxis]) / target.shape[1] + mean_derivative[-1] / target.shape[1]       
        delta[-1] = net_derivative[-1]
       
        for i in reversed(range(self.num_layers - 1)):
            if i > 0:
                y_derivative[i] = np.dot(self.weights[i].T, delta[i + 1]) * self.relu_derivative(out_forward[0][i])
            else:
                y_derivative[i] = np.dot(self.weights[i].T, delta[i + 1])
            x_norm_derivative[i] = y_derivative[i] * self.gamma[i]
            var_derivative[i] = (x_norm_derivative[i] * (out_forward[1][i] - out_forward[3][i][:, 0][:, np.newaxis]) * \
                                 (-0.5) * pow(out_forward[3][i][:, 1][:, np.newaxis] + self.epsilon, -1.5)).sum(axis=1)[:, np.newaxis]
            mean_derivative[i] = (x_norm_derivative[i] * (-1) / np.sqrt(out_forward[3][i][:, 1][:, np.newaxis] + self.epsilon)).sum(axis=1)[:, np.newaxis] \
                                 + var_derivative[i] * (-2 * (out_forward[1][i] - out_forward[3][i][:, 0][:, np.newaxis])).sum(axis=1)[:, np.newaxis] / target.shape[1]
            net_derivative[i] = x_norm_derivative[i] / np.sqrt(out_forward[3][i][:, 1][:, np.newaxis] + self.epsilon) + \
                                var_derivative[i] * 2 * (out_forward[1][i] - out_forward[3][i][:, 0][:, np.newaxis]) / \
                                target.shape[1] + mean_derivative[i] / target.shape[1]
            delta[i] = net_derivative[i]

      
        lambda_weights = [np.dot(delta[i + 1], out_forward[0][i].T) for i in range(self.num_layers - 1)]       
        lambda_gamma = [(yd * xm).sum(axis=1)[:, np.newaxis] for yd, xm in zip(y_derivative, out_forward[2])]      
        lambda_beta = [yd.sum(axis=1)[:, np.newaxis] for yd in y_derivative]
      
        self.weights = [w - eta * (slw/target.shape[1]) for w, slw in zip(self.weights,lambda_weights)]       
        self.gamma = [g - eta * lg for g, lg in zip(self.gamma, lambda_gamma)]
        self.beta = [be - eta * lbe for be, lbe in zip(self.beta, lambda_beta)]
       
        back_delta = np.zeros(inpt_array.shape)
        for col in range(delta[0].shape[1]):
            back_delta[col] = delta[0][:,col].reshape(inpt_array.shape[1], inpt_array.shape[2], inpt_array.shape[3])
        return back_delta

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        y = np.where(x > 0, x ,0.01 * x)
        return y

    def relu_derivative(self, x):
        y = np.where(x > 0, 1, 0.01)
        return y

    def cost_derivative(self, out, target):
        return (out - target)

    def cost(self, out, target):
        return  sum(0.5 * sum((out - target) ** 2))

    def softmax(self, net):
        return np.exp(net)/sum(np.exp(net))

    def softmax_derivative(self, out):
        return out * (1 - out)

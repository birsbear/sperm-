# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:26:39 2018

@author: Brian.Chiu
"""

#load .mat data __ data preprocess

import scipy, time, cv2, os
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd
from mxnet import init
from mxnet import image
from mxnet.gluon.model_zoo import vision as models
from mxboard import SummaryWriter
import utils

def get_transform(augs):
    def transform(data, label):
        # data: sample x height x width x channel
        # label: sample
        data = data.astype('float32')
        if augs is not None:
            # apply to each sample one-by-one and then stack
            data = nd.stack(*[
                apply_aug_list(d, augs) for d in data])
        data = nd.transpose(data, (0,3,1,2))
        return data, label.astype('float32')
    return transform


def image_aug_transform(data,label):
    img_augs = [image.HorizontalFlipAug(1)]
    array_rate = 8
    data_array = np.zeros((data.shape[0]*array_rate,data.shape[1],data.shape[2]))
    label_array = np.ones(data_array.shape[0])*label
    data_array[0:data.shape[0],:,:] = data
    for i, d in enumerate(data):
        for j in img_augs:
            for k in range(1,array_rate):
                if k%2 == 1:
                    data_array[data.shape[0]*k+i,:,:] = j(nd.array(data_array[data.shape[0]*(k-1)+i,:,:])).asnumpy()
                else:
                    data_array[data.shape[0]*k+i,:,:] = data_array[data.shape[0]*(k-1)+i,:,:].T
                
        
    
    return data_array, label_array

def data_class_process(pre_data,pre_label,data_num,train_augs,class_value = [0,3,4],augs_value = [1,2]):
    new_label = np.zeros((0,0))
    new_data = np.zeros((0,128,128))
    for i, v in enumerate(class_value):
        if i <len(class_value)-1:
            for j in range(class_value[i+1]-class_value[i]):
                pre_label[np.where(pre_label == v+j)] = i
            index = np.random.permutation(np.where(pre_label == i)[0])
            if i in augs_value:
                data_array, label_array = image_aug_transform(pre_data[index],i)  #把data增廣
                r_index = np.random.permutation(range(len(data_array)))
                new_label = np.append(new_label,label_array[r_index[0:data_num[i]]])
                new_data = np.append(new_data,data_array[r_index[0:data_num[i]]],axis = 0)
            else:
                new_label = np.append(new_label,pre_label[index[0:data_num[i]]])
                new_data = np.append(new_data,pre_data[index[0:data_num[i]]],axis = 0)            
            
        else:
            pre_label[np.where(pre_label >= v)] = i
            index = np.random.permutation(np.where(pre_label == i)[0])
            if i in augs_value:
                data_array, label_array = image_aug_transform(pre_data[index],i)
                r_index = np.random.permutation(range(len(data_array)))
                new_label = np.append(new_label,label_array[r_index[0:data_num[i]]])
                new_data = np.append(new_data,data_array[r_index[0:data_num[i]]],axis = 0)
            else:
                new_label = np.append(new_label,pre_label[index[0:data_num[i]]])
                new_data = np.append(new_data,pre_data[index[0:data_num[i]]],axis = 0)  
            
            
    return pre_data, pre_label,new_label, new_data
                
def get_data(file_path,rand_seed,batch_size,train_augs):
    #從 mat 檔獲得 data資訊
    sperm_data = scipy.io.loadmat(file_path)['cnn_data']
    pre_data = []
    pre_label = []
    for i in sperm_data:
        for j in i[0]:
            pre_data.append(np.float64(j[0]))
            pre_label.append(np.int32(j[1][0][0]))
#            tic = time.time()
#            pre_data = np.append(pre_data,nd.array(cv2.resize(j[0],(94,94))).expand_dims(axis=0).asnumpy(),axis = 0)
#            pre_label = np.append(pre_label,np.int32(j[1][0][0]))
#            print(time.time()-tic)
    pre_data = np.asarray([cv2.resize(pre_data,(128,128)) for pre_data in pre_data])
    pre_data = pre_data.astype(np.float32, copy=False)/(255.0/2) - 1.0
    pre_label = np.array(pre_label)
    
    class_value = [0,3,4]  #所需要流下的種類編號
    data_num = [1536,1536,1024] #
    pre_data, pre_label,new_label, new_data = data_class_process(pre_data,pre_label,data_num,train_augs,class_value)
#    plt.imshow(pre_data[0])
    pre_data = nd.array(new_data).expand_dims(axis= 1)
    pre_data = nd.tile(pre_data,(1,3,1,1))
    
    pre_label = np.array(new_label)
    np.random.seed(rand_seed)
    rand_index = np.random.permutation(np.shape(pre_data)[0])
    
#    plt.figure(2)
#    plt.imshow(((pre_data[0][0]+1)*(255/2)).asnumpy())
    
    pre_train = [pre_data[rand_index[0:-np.int32(rand_index.shape[0]/4)]],pre_label[rand_index[0:-np.int32(rand_index.shape[0]/4)]]]
    pre_test = [pre_data[rand_index[-np.int32(rand_index.shape[0]/4):]],pre_label[rand_index[-np.int32(rand_index.shape[0]/4):]]]
    train_iter = mx.io.NDArrayIter(data = pre_train[0], label= pre_train[1], batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(data = pre_test[0], label= pre_test[1], batch_size=batch_size)
    
    return train_iter, test_iter, data_num
    
def apply_aug_list(img, augs):
    for f in augs:
        img = f(img)
    return img

def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    acc = nd.array([0])
    n = 0.
    data_iterator.reset()
    for batch in data_iterator:
        data = batch.data
        label = batch.label
        for X, y in zip(data, label):
            y = y.astype('float32')
            acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n

def show_evaluate_value(data_iterator, net, data_num,code_path,result_path, ctx = mx.gpu(0)):
    data_iterator.reset()
    testv = data_iterator.label[0][1]
    netv = nd.zeros(testv.shape[0])
    true_num = nd.zeros((3,))
    fail_test = []
    fail_net = []
    error_list = []
    for i, batch in enumerate(data_iterator):
        data = batch.data
        label = batch.label
        for X, y in zip(data, label):
            ty = net(X).argmax(axis = 1)
#            testv[i*64:i*64+64] = y
            netv[i*64:i*64+64] = ty
            for ind,n in enumerate(ty):
                if n == y[ind]:
                    true_num[n] += 1
                else:
                    fail_test.append([i,ind,y[ind].asnumpy()[0]])
                    fail_net.append([i,ind,n.asnumpy()[0]])
                    error_list.append(y[ind].asnumpy()[0]*10+n.asnumpy()[0])
#                    plt.figure((i*64+ind)*10+np.int32(y[ind].asnumpy()[0]))
                    plt.figure(1)
                    plt.imshow((X[ind].transpose((1,2,0))+125)[:,:,0].asnumpy(),cmap='gray')
                    os.chdir(result_path)
                    plt.savefig('image%d_%d'% ((i*64+ind)*10+np.int32(y[ind].asnumpy()[0]),n.asnumpy()))
                    os.chdir(code_path)
                    plt.close()
#            for i in range(len(ty))
#            print(netv)
#            print('No.%d net value ='%i,ty)
#            print('No.%d test value ='% i,y)
    print('Abnormal data num: %d , data rate: %.2f%%, Abnormal acc : %.2f%% \n' % (list(testv).count(0), list(testv).count(0)/(data_num[0])*100, (true_num[0]/list(testv).count(0)*100).asnumpy()[0]))
    print('Normal data num: %d , data rate: %.2f%%, Normal acc : %.2f%% \n' % (list(testv).count(1),  list(testv).count(1)/(data_num[1])*100, (true_num[1]/list(testv).count(1)*100).asnumpy()[0]))
    print('Not Sperm data num: %d , data rate: %.2f%%, Abnormal acc : %.2f%% \n' % (list(testv).count(2), list(testv).count(2)/(data_num[2])*100, (true_num[2]/list(testv).count(2)*100).asnumpy()[0]))                    
#    print('Abnormal data num: %d , Abnormal acc : %.2f%%' % (list(testv).count(0), (true_num[0]/list(testv).count(0)*100).asnumpy()[0]))
#    print('Normal data num: %d , Normal acc : %.2f%%' % (list(testv).count(1), (true_num[1]/list(testv).count(1)*100).asnumpy()[0]))
#    print('Not Sperm data num: %d , Not Sperm acc : %.2f%%' % (list(testv).count(2), (true_num[2]/list(testv).count(2)*100).asnumpy()[0]))
    print('Error 0->1 : %d' % error_list.count(1))
    print('Error 0->2 : %d' % error_list.count(2))
    print('Error 1->0 : %d' % error_list.count(10))
    print('Error 1->2 : %d' % error_list.count(12))
    print('Error 2->0 : %d' % error_list.count(20))
    print('Error 2->1 : %d' % error_list.count(21))
    return testv, netv, true_num, fail_test, fail_net,error_list
            
def test_evaluate(data_path,net):
    sperm_d = scipy.io.loadmat(data_path)['cnn_data']
    t_data = []
    for i in sperm_d:
        for j in i[0]:
            t_data.append(np.float64(j[0]))
            
    t_data = np.asarray([cv2.resize(t_data,(128,128)) for t_data in t_data])
    t_data = t_data.astype(np.float32, copy=False)/(255.0/2) - 1.0
    t_data = nd.array(t_data).expand_dims(axis = 1)
    t_data = nd.tile(t_data,(1,3,1,1))
            
    t_iter = mx.io.NDArrayIter(data = t_data)
                
    t_iter.reset()
    y = nd.zeros(len(t_iter.data[0][1]))
    for i, batch in enumerate(t_iter):
        data = batch.data
        for X in data:
            y[i] = net(X).argmax(axis=1)
                    
    return y
                
def rescale(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min().asscalar()
    if x_max is None:
        x_max = x.max().asscalar()
    return (x - x_min) / (x_max - x_min)


def rescale_per_image(x):
    assert x.ndim == 4
    x = x.copy()
    for i in range(x.shape[0]):
        min_val = x[i].min().asscalar()
        max_val = x[i].max().asscalar()
        x[i] = rescale(x[i], min_val, max_val)
    return x

train_augs = image.HorizontalFlipAug(1)

test_augs = [
    image.CenterCropAug((32,32))
]



batch_size = 64
learning_rate=.1
num_epochs = 500
randseed = 10
test_n = 'test15'
ctx = utils.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_data, test_data, data_num = get_data('C:/lagBear/SEMEN/finally_data/cnn_data.mat',randseed,batch_size=batch_size,train_augs=train_augs)

### resnet network ###
pre_net = models.resnet50_v1(pretrained=True,prefix = 'sperm_3class_')
pre_net.output
pre_net.features[0].weight.data()[0][0]

net = models.resnet50_v1(classes=3,prefix = 'sperm_3class_')
net.features = pre_net.features
net.output.initialize(init.Xavier())
net.hybridize()
sw = SummaryWriter(logdir = './logs/resnet50/randseed%s/%s' % (randseed,test_n) ,flush_secs = 5)





trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': learning_rate})
#    utils.train(
#        train_data, test_data, net, loss, trainer, ctx, num_epochs)    
#    #######
print("Start training on ", ctx)
print_batches=None

if not os.path.isdir('spermdata'):
    os.mkdir('spermdata')
if not os.path.isdir('spermdata/fix3c/%s' % (test_n)):
    os.mkdir('spermdata/fix3c/%s' % (test_n))

code_path = os.getcwd()
result_path = 'C:\\code\\spermdata\\fix3c\\%s' %(test_n)



for epoch in range(num_epochs):
    train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
    start = time.time()
    train_data.reset()
    
    for i, batch in enumerate(train_data):
        data = batch.data
        label = batch.label
        losses = []
        with autograd.record():
            outputs = [net(X) for X in data]
            losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
        for l in losses:
            l.backward()
        train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                          for yhat, y in zip(outputs, label)])
        train_loss += sum([l.sum().asscalar() for l in losses])
        trainer.step(batch_size)
        n += batch_size
        m += sum([y.size for y in label])
        if print_batches and (i+1) % print_batches == 0:
            
            print("Batch %d. Loss: %f, Train acc %f" % (
                n, train_loss/n, train_acc/m
            ))
            
### 可視化 ###            
        if i == 0:
            data_image = data[0]
            data_image = (data_image-data_image.min())/(data_image.max()-data_image.min())
            sw.add_image('sperm_image',data_image,epoch)
    if epoch == 0:
        sw.add_graph(net)
### 可視化 ###        
        
    test_acc = evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
        epoch, train_loss/n, train_acc/m, test_acc, time.time() - start
    ))
    
        
    if train_loss/n <= 0.0008 and epoch >= 20:
        break    
### 可視化 ###
    pn = list(net.collect_params().keys())
    param_names, grads = [], []
    for n,i in enumerate(net.collect_params().values()):
        if i.grad_req != 'null':
            grads.append(i.grad())
            param_names.append(pn[n])
    assert len(grads) == len(param_names)
    # logging the gradients of parameters for checking convergence
    for i, name in enumerate(param_names):
        sw.add_histogram(tag = name, values = grads[i], global_step = epoch, bins = 1000)
    sw.add_scalar(tag='cross_entropy', value=train_loss/n, global_step=epoch)
    sw.add_scalar(tag= 'train_acc', value = train_acc/m*100, global_step=epoch)
    sw.add_scalar(tag= 'test_acc', value = test_acc*100, global_step=epoch)

test_data.reset()
filter_image = test_data.data[0][1][0:2]
sw.add_image(tag = 'int_put_test_image',image= rescale_per_image(filter_image))

L1conv_net = gluon.nn.Sequential()
L1conv_net.add(net.features[0])
o_f_image = L1conv_net(filter_image)
o_f_image = o_f_image[0:1].transpose((1,0,2,3))

sw.add_image(tag = 'out_image', image = rescale_per_image(o_f_image))
sw.add_image(tag = 'weight_image', image=rescale_per_image(L1conv_net[0].weight.grad()))

### 可視化 ###
sw.close()
print('2 Data Aug Sperm 3 class predict \n%s, Rand seed %d, test acc %.2f%%'% (test_n, randseed, test_acc*100))
tv,nv,tn,ft,fn,el = show_evaluate_value(test_data,net,data_num,code_path,result_path,ctx = ctx)
with open('2_data_Sperm_3class_result.txt','a+') as f :
    f.write('Test 15, Rand seed %d\n\n' % randseed)
    f.write('2 Data Aug Sperm 3 class predict, Rand seed %d, test acc %.2f%%\n'% (randseed, test_acc*100))
    f.write('Abnormal data num: %d , data rate: %.2f%%, Abnormal acc : %.2f%% \n' % (list(tv).count(0), list(tv).count(0)/(data_num[0])*100, (tn[0]/list(tv).count(0)*100).asnumpy()[0]))
    f.write('Normal data num: %d , data rate: %.2f%%, Normal acc : %.2f%% \n' % (list(tv).count(1),  list(tv).count(1)/(data_num[1])*100, (tn[1]/list(tv).count(1)*100).asnumpy()[0]))
    f.write('Not Sperm data num: %d , data rate: %.2f%%, Abnormal acc : %.2f%% \n' % (list(tv).count(2), list(tv).count(2)/(data_num[2])*100, (tn[2]/list(tv).count(2)*100).asnumpy()[0]))
    f.write('Error 0->1 : %d\t' % el.count(1))
    f.write('Error 1->0 : %d\t' % el.count(10))
    f.write('Error 2->0 : %d\t\n' % el.count(20))
    f.write('Error 0->2 : %d\t' % el.count(2))
    f.write('Error 1->2 : %d\t' % el.count(12))
    f.write('Error 2->1 : %d\t\n\n\n' % el.count(21))    
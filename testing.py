# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
import sklearn
from sklearn.manifold import TSNE
import torch
import pandas as pd
from sklearn import preprocessing
from torch.autograd import Variable
import itertools
from sklearn  import metrics
from tqdm import tqdm

from dataset1d import Dataset

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) for k in topk]

def my_plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Greens, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('spring')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def my_plot_confusion_matrix2(cm, savename, classes, title='Confusion Matrix'):

    plt.figure(figsize=(8, 8), dpi=100)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c >= 0.:
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGn)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.show()


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    #ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i]+1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    return fig

def get_data2(path,data_root):
    testset = Dataset(root=data_root, train=False,transform=None) 
    #testset = Dataset(root='KDDTest_one_hot_-21.csv', train=False,transform=None) 
    # samples_weight = torch.load('checkpoint/samples_weight.h5')
    # samples_weight = np.load('sample_w.npy')
    # samplers = WeightedRandomSampler(samples_weight,len(trainset),replacement=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=4)
    net = torch.load(path)['net']
    # net = torch.load('checkpoint/hlstm_ckpt.t7')['net']
    #net = torch.load('checkpoint3/dnn_87.66_ckpt.t7')['net']
    #net = torch.load('checkpoint/lstm_ckpt.t7')['net']
    #net = torch.load('checkpoint/cnn_10_20_78.14_ckpt.t7')['net']

    predicts,labels =[],[]
    acc1 = 0
    correct=0
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()
        top1, _ = accuracy(outputs, targets, topk=(1, 5))
        acc1 += top1.item()

        predicts.append(outputs.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())

    accur = 100. * acc1 / len(testset)
    print('accuracy =',accur)
    predicts = np.concatenate(predicts)
    labels = np.concatenate(labels)

    return predicts, labels



def get_data():

    df = pd.read_csv('NSL_KDD_Test_one_hot.csv', header=None)
    #df = pd.concat([df[df.iloc[:,121]==i].sample(70000,replace=True) for i in range(5)])

    data = df.iloc[:, 0:121].to_numpy()
    data = preprocessing.scale(data)
    data = Variable(torch.FloatTensor(data).cuda())

    labels = df.iloc[:, 121].to_numpy()


    checkpoint = torch.load('./checkpoint3/dnn_87.66_ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc1']
    print('best acc= ',best_acc)
    # net = torch.load('./checkpoint/ckpt.t7')['net']
    # print(net)

    y_data = net(data).detach().cpu().numpy()

    return y_data, labels

def get_one_hot(predicts,ids):
    one_hot_predicts=np.zeros(predicts.shape)
    for i in range(len(predicts)):
        one_hot_predicts[i,ids[i]] = 1
    return one_hot_predicts

def main(predicts, y_labels):
    
    df_k = pd.read_csv('KDDTest+.txt',header=None)
    print(len(df_k))
    
    ### add the following code for NSL-KDD Test-21 testing.
    df_k = df_k.to_numpy()
    df_index = df_k[:,-1]!=21
    predicts = predicts[df_index]
    y_labels = y_labels[df_index]
    
    
    print(len(predicts))
    y_pred = np.argmax(predicts, axis=-1)
    conf_mat = metrics.confusion_matrix(y_true=y_labels, y_pred=y_pred)
    result_info = metrics.classification_report(y_labels,y_pred,digits=4)

    print(conf_mat)
    print(result_info)
    
    #plt.figure()
    labels = ['normal', 'DoS', 'Probe', 'U2R', 'R2L']
    my_plot_confusion_matrix2(conf_mat, 'con_mat+_df_85.64.png', labels)

    acc = metrics.accuracy_score(y_pred,y_labels) * 100
    f1 = metrics.f1_score(y_labels,y_pred,average='weighted') * 100
    r = metrics.recall_score(y_labels,y_pred,average='weighted') * 100
    p = metrics.precision_score(y_labels,y_pred,average='weighted') * 100
    #pr_curve = metrics.precision_recall_curve(y_labels,predicts) 
    
    # rc = roc_curve(y_labels, predicts)
    print('acc={:.2f}, f1={:.2f},r={:.2f}, p={:.2f}'.format(acc,f1,r,p))

    

if __name__ == '__main__':
    # KDDTest_one_hot_-21.csv
    predicts, y_labels = get_data2('checkpoint/df_85.64_ckpt.t7','NSL_KDD_Test_one_hot.csv') # 
    main(predicts, y_labels)

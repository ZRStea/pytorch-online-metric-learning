import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import argparse
import pandas as pd
import random
import os
from glob import glob
import time
from torchvision import models, transforms
from torch.utils.data import DataLoader
from layers import L2Normalization
from torch.autograd import Variable
from losses import *
from scipy.io import loadmat
def logfunc(log_name, message):
    print(message)
    with open(log_name, 'a') as log_file:
        log_file.write(message)
        log_file.write('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',required=True, help='path of dataset')
parser.add_argument('--cuda',default=True, action='store_true', help='cuda')
parser.add_argument('--batch_size', type=int, required=True, help='batchsize for train')
parser.add_argument('--batch_size_test', type=int, required=True, help='batchsize for test')
parser.add_argument('--eval_interval', type=int, default=500, help='evaluation iteration interval')
parser.add_argument('--embedding_dim', type=int, required=True, help='embedding dim')
parser.add_argument('--loss', required=True, help='loss type')
parser.add_argument('--mining', help='triplet mining strategy, \'hardest\' or \'semi\' or \'all\' ')
parser.add_argument('--dataset', required=True, help='dataset name')
parser.add_argument('--decay_iter_step', type=int, required=True, help='decay_iter_step')
parser.add_argument('--seed', type=int, required=True, help='random seed')
parser.add_argument('--decay_gamma', type=float, required=True, help='decay_gamma')
parser.add_argument('--checkpoints_path', default='.', help='path of saving model and log')
parser.add_argument('--margin', type=float,required=True, help='loss margin')
parser.add_argument('--nworkers', default=10, type=int, help='number of loading workers')
parser.add_argument('--lr', type=float, required=True,default=1e-4, help='learning rate')
parser.add_argument('--maxiter', type=int, required=True, default=10000, help='maxiter')
parser.add_argument('--l2norm',default=False, action='store_true', help='enables l2norm')
opt = parser.parse_args()
random.seed(opt.seed)
torch.manual_seed(opt.seed)
try:
    os.makedirs(opt.checkpoints_path)
except OSError:
    pass

if opt.dataset == 'CUB-200-2011':
    from datasets_cub import ImageDatasetTest, ImageDatasetTrain
    from samplers_cub import CUBSampler
    from evaluation_cub import Evaluation
    #---------Read images id, path, label-----------
    img_paths = pd.read_csv(os.path.join(opt.dataroot, 'images.txt'), sep=" ", header=None)
    img_paths.columns = ["id", "path"]
    labels = pd.read_csv(os.path.join(opt.dataroot, 'image_class_labels.txt'), sep=" ", header=None)
    labels.columns = ["id", "label"]
    is_train = pd.read_csv(os.path.join(opt.dataroot, 'train_test_split.txt'), sep=" ", header=None)
    is_train.columns = ["id", "is_train"]
    data = pd.concat([img_paths,labels["label"],is_train["is_train"]],axis=1)
    #---------------change path---------------------
    paths = data["path"].tolist()
    paths = [os.path.join(opt.dataroot,'images',i) for i in paths]
    data["path"] = paths
    #---------------split dataset-------------------
    data_train = data[data["label"] <= 100]
    data_test = data[data["label"] > 100]

    data_transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_test = ImageDatasetTest(data_test['path'],transform=data_transform_test)
    dataloader_test =  DataLoader(dataset_test, batch_size=opt.batch_size_test, shuffle=False, num_workers=opt.nworkers)

    data_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageDatasetTrain(data_train['path'], data_train['label'], data_transform)
    sampler = CUBSampler(data_train['label'], opt.batch_size, opt.maxiter)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=opt.nworkers)

    evaluation = Evaluation(data_test, data_test, dataloader_test, dataloader_test, opt.cuda)

if opt.dataset == 'In-Shop':
    from datasets_shop import ImageDatasetTest, ImageDatasetTrain
    from samplers_shop import ShopSampler
    from evaluation_shop import Evaluation
    #---------Read images id, path, label------
    data = pd.read_csv(os.path.join(opt.dataroot,"Eval",'list_eval_partition.txt'), sep = "\s+|\t+|\s+\t+|\t+\s+")
    data.columns = ["path","label","evaluation_status"]
    #--------------change path---------------------
    paths = data["path"].tolist()
    paths = [os.path.join(opt.dataroot,i) for i in paths]
    data["path"] = paths
    #-----------------change label-----------------
    labels = data["label"].tolist()
    labels = [int(i[3:]) for i in labels]
    data["label"] = labels
    #-----------------------------------------
    data_train = data[data["evaluation_status"] == "train"]
    data_query = data[data["evaluation_status"] == "query"]
    data_gallery = data[data["evaluation_status"] == "gallery"]

    data_transform_query = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_query = ImageDatasetTest(data_query['path'],transform=data_transform_query)
    dataloader_query =  DataLoader(dataset_query, batch_size=opt.batch_size_test, shuffle=False, num_workers=opt.nworkers)
    
    data_transform_gallery = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_gallery = ImageDatasetTest(data_gallery['path'],transform=data_transform_gallery)
    dataloader_gallery =  DataLoader(dataset_gallery, batch_size=opt.batch_size_test, shuffle=False, num_workers=opt.nworkers)

    data_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageDatasetTrain(data_train['path'], data_train['label'], data_transform)
    sampler = ShopSampler(data_train['label'], opt.batch_size, opt.maxiter)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=opt.nworkers)

    evaluation = Evaluation(data_gallery, data_query, dataloader_gallery, dataloader_query, opt.cuda)

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential()
model.fc.add_module('fc', nn.Linear(num_ftrs, opt.embedding_dim))
if opt.l2norm:
    model.fc.add_module('l2normalization', L2Normalization())
if opt.cuda:
    model = model.cuda()

if opt.loss == 'lifted':
    criterion = LiftedLoss(margin = opt.margin, cuda=opt.cuda)
if opt.loss == 'triplet':
    criterion = TripletLoss(margin = opt.margin, cuda=opt.cuda, mining=opt.mining)
if opt.loss == 'contrastive':
    criterion = ContrastiveLoss(margin = opt.margin, cuda=opt.cuda)


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.decay_iter_step, gamma=opt.decay_gamma)

start_time = time.time()
log_name = os.path.join(opt.checkpoints_path, 'loss_log.txt')

logfunc(log_name, 'Training {}'.format(time.strftime('%c')))
logfunc(log_name, str(opt))
#---------------------------------------------------
iter_num = 0
running_loss = .0
model.train(False)
ranks, mAP , nmi = evaluation.ranks_map(model)
if opt.dataset == 'In-Shop':
    message = '[iter {}/{}] Time elapsed: {:.4f}; Rank1: {:.4f}, Rank10: {:.4f},Rank20: {:.4f},Rank30: {:.4f},Rank40: {:.4f},mAP: {:.4f},NMI: {:.4f},loss: {:.6f}'.format(iter_num, opt.maxiter, time.time() - start_time, ranks[0],ranks[9],ranks[19],ranks[29],ranks[39], mAP, nmi, running_loss/opt.eval_interval)
if opt.dataset == 'CUB-200-2011' or opt.dataset == 'Cars196':
    message = '[iter {}/{}] Time elapsed: {:.2f}; Rank1: {:.4f}, Rank2: {:.4f},Rank4: {:.4f},Rank8: {:.4f},Rank16: {:.4f},Rank32: {:.4f},mAP: {:.4f},NMI: {:.4f},loss: {:.6f}'.format(iter_num, opt.maxiter, time.time() - start_time, ranks[0],ranks[1],ranks[3],ranks[7],ranks[15],ranks[31], mAP ,nmi,running_loss/opt.eval_interval)
logfunc(log_name, message)

for data in dataloader:
    iter_num += 1
    scheduler.step()
    model.train(True)

    inputs, labels = data
    inputs, labels = Variable(inputs.squeeze()), labels.squeeze()

    if opt.cuda:
        inputs, labels = inputs.cuda(), labels.cuda()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    running_loss += loss.data[0]

    if iter_num % opt.eval_interval == 0:
        model.train(False)
        ranks, mAP , nmi = evaluation.ranks_map(model)
        if opt.dataset == 'In-Shop':
            message = '[iter {}/{}] Time elapsed: {:.4f}; Rank1: {:.4f}, Rank10: {:.4f},Rank20: {:.4f},Rank30: {:.4f},Rank40: {:.4f},mAP: {:.4f},NMI: {:.4f},loss: {:.6f}'.format(iter_num, opt.maxiter, time.time() - start_time, ranks[0],ranks[9],ranks[19],ranks[29],ranks[39], mAP, nmi, running_loss/opt.eval_interval)
        if opt.dataset == 'CUB-200-2011' or opt.dataset == 'Cars196':
            message = '[iter {}/{}] Time elapsed: {:.2f}; Rank1: {:.4f}, Rank2: {:.4f},Rank4: {:.4f},Rank8: {:.4f},Rank16: {:.4f},Rank32: {:.4f},mAP: {:.4f},NMI: {:.4f},loss: {:.6f}'.format(iter_num, opt.maxiter, time.time() - start_time, ranks[0],ranks[1],ranks[3],ranks[7],ranks[15],ranks[31], mAP ,nmi,running_loss/opt.eval_interval)
        logfunc(log_name, message)
        if iter_num >= opt.maxiter and opt.dataset == 'Online_Product':
            ranks, mAP , nmi = evaluation.ranks_map(model,nmi_enbale=True)
            message = '[iter {}/{}] Time elapsed: {:.4f}; Rank1: {:.4f}, Rank10: {:.4f},Rank100: {:.4f},Rank1000: {:.4f},mAP: {:.4f},NMI: {:.4f},loss: {:.6f}'.format(iter_num, opt.maxiter, time.time() - start_time, ranks[0],ranks[9],ranks[99],ranks[999], mAP, nmi, running_loss/opt.eval_interval)
            logfunc(log_name, message)
        running_loss = .0

torch.save(model.state_dict(), os.path.join(opt.checkpoints_path, 'model_params.pth'))
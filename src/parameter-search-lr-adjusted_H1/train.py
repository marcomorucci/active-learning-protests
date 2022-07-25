"""
created by: Donghyeon Won
Modified codes from
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://github.com/pytorch/examples/tree/master/imagenet
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
#from itertools import ifilter
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

from util import ProtestDataset, modified_resnet50, AverageMeter, Lighting,ProtestDataset_AL
#from pred import eval_one_data

# for indexing output of the model
protest_idx = Variable(torch.LongTensor([0]))
violence_idx = Variable(torch.LongTensor([1]))
visattr_idx = Variable(torch.LongTensor(range(2,12)))
best_loss = float("inf")

def calculate_loss(output, target, criterions, weights = [1, 10, 5]):
    """Calculate loss"""
    # number of protest images
    N_protest = int(target['protest'].data.sum())
    batch_size = len(target['protest'])

    if N_protest == 0:
        # if no protest image in target
        outputs = [None]
        # protest output
        outputs[0] = output.index_select(1, protest_idx)
        targets = [None]
        # protest target
        targets[0] = target['protest'].float()
        losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(1)]
        scores = {}
        scores['protest_acc'] = accuracy_score((outputs[0]).data.cpu().round(), targets[0].data.cpu())
        scores['violence_mse'] = 0
        scores['visattr_acc'] = 0
        return losses, scores, N_protest

    # used for filling 0 for non-protest images
    not_protest_mask = (1 - target['protest']).bool()

    outputs = [None] * 4
    # protest output
    outputs[0] = output.index_select(1, protest_idx)
    # violence output
    outputs[1] = output.index_select(1, violence_idx)
    outputs[1].masked_fill_(not_protest_mask, 0)
    # visual attribute output
    outputs[2] = output.index_select(1, visattr_idx)
    outputs[2].masked_fill_(not_protest_mask.repeat(1, 10),0)


    targets = [None] * 4

    targets[0] = target['protest'].float()
    targets[1] = target['violence'].float()
    targets[2] = target['visattr'].float()

    scores = {}
    # protest accuracy for this batch
    scores['protest_acc'] = accuracy_score(outputs[0].data.cpu().round(), targets[0].data.cpu())
    # violence MSE for this batch
    scores['violence_mse'] = ((outputs[1].data - targets[1].data).pow(2)).sum() / float(N_protest)
    # mean accuracy for visual attribute for this batch
    comparison = (outputs[2].data.round() == targets[2].data)
    comparison.masked_fill_(not_protest_mask.repeat(1, 10).data,0)
    n_right = comparison.float().sum()
    mean_acc = n_right / float(N_protest*10)
    scores['visattr_acc'] = mean_acc

    # return weighted loss
    losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(len(criterions))]

    return losses, scores, N_protest



def train(train_loader, model, criterions, optimizer, epoch):
    """training the model"""

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_protest = AverageMeter()
    loss_v = AverageMeter()
    protest_acc = AverageMeter()
    violence_mse = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(train_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        input_var = Variable(input)
        output = model(input_var)

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)

        optimizer.zero_grad()
        loss = 0
        for l in losses:
            loss += l
        # back prop
        loss.backward()
        optimizer.step()

        if N_protest:
            loss_protest.update(losses[0].data, input.size(0))
            loss_v.update(loss.data - losses[0].data, N_protest)
        else:
            # when there is no protest image in the batch
            loss_protest.update(losses[0].data, input.size(0))
        loss_history.append(loss.data)
        protest_acc.update(scores['protest_acc'], input.size(0))
        violence_mse.update(scores['violence_mse'], N_protest)
        visattr_acc.update(scores['visattr_acc'], N_protest)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'Protest {protest_acc.val:.3f} ({protest_acc.avg:.3f})  '
                  'Violence {violence_mse.val:.5f} ({violence_mse.avg:.5f})  '
                  'Vis Attr {visattr_acc.val:.3f} ({visattr_acc.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   loss_val=loss_protest.val + loss_v.val,
                   loss_avg = loss_protest.avg + loss_v.avg,
                   protest_acc = protest_acc, violence_mse = violence_mse,
                   visattr_acc = visattr_acc))

    return loss_history

def validate(val_loader, model, criterions, epoch):
    """Validating"""
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_protest = AverageMeter()
    loss_v = AverageMeter()
    protest_acc = AverageMeter()
    violence_mse = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    loss_history = []
    for i, sample in enumerate(val_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        input_var = Variable(input)

        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        output = model(input_var)

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)
        loss = 0
        for l in losses:
            loss += l

        if N_protest:
            loss_protest.update(losses[0].data, input.size(0))
            loss_v.update(loss.data - losses[0].data, N_protest)
        else:
            # when no protest images
            loss_protest.update(losses[0].data, input.size(0))
        loss_history.append(loss.data)
        protest_acc.update(scores['protest_acc'], input.size(0))
        violence_mse.update(scores['violence_mse'], N_protest)
        visattr_acc.update(scores['visattr_acc'], N_protest)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Loss {loss_val:.3f} ({loss_avg:.3f})  '
                  'Protest Acc {protest_acc.val:.3f} ({protest_acc.avg:.3f})  '
                  'Violence MSE {violence_mse.val:.5f} ({violence_mse.avg:.5f})  '
                  'Vis Attr Acc {visattr_acc.val:.3f} ({visattr_acc.avg:.3f})'
                  .format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   loss_val =loss_protest.val + loss_v.val,
                   loss_avg = loss_protest.avg + loss_v.avg,
                   protest_acc = protest_acc,
                   violence_mse = violence_mse, visattr_acc = visattr_acc))

    print(' * Loss {loss_avg:.3f} Protest Acc {protest_acc.avg:.3f} '
          'Violence MSE {violence_mse.avg:.5f} '
          'Vis Attr Acc {visattr_acc.avg:.3f} '
          .format(loss_avg = loss_protest.avg + loss_v.avg,
                  protest_acc = protest_acc,
                  violence_mse = violence_mse, visattr_acc = visattr_acc))
    return loss_protest.avg + loss_v.avg, loss_history

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every 5 epochs"""
    lr = args.lr * (0.4 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# def adjust_learning_rate_dynamic(optimizer, samples):
#     """Sets the learning rate to the initial LR decayed by 0.5 every 5 epochs"""
#     if samples > 800:
#         lr = args.lr * (800/samples)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',filename_best_model='model_best.pth.tar'):
    """Save checkpoints"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename_best_model)

def eval_one_data(img_df, img_dir, model, n):
        """
        return model output of all the images in a directory
        """
        model.eval()
        # make dataloader
        dataset = ProtestDataset_AL(img_dir = img_dir, img_df = img_df)
        data_loader = DataLoader(dataset,
                                num_workers = args.workers,
                                batch_size = args.batch_size)
        # load model

        outputs = []
        imgpaths = []

        n_imgs = len(img_df.iloc[:,0]) #len(os.listdir(img_dir))
        with tqdm(total=n_imgs) as pbar:
            for i, sample in enumerate(data_loader):
                imgpath, input = sample['imgpath'], sample['image']
                if args.cuda:
                    input = input.cuda()

                input_var = Variable(input)
                output = model(input_var)
                outputs.append(output.cpu().data.numpy())
                imgpaths += imgpath
                if i < n_imgs / args.batch_size:
                    pbar.update(args.batch_size)
                else:
                    pbar.update(n_imgs%args.batch_size)


        df = pd.DataFrame(np.zeros((n_imgs, 13)))
        df.columns = ["imgpath", "protest", "violence", "sign", "photo",
                      "fire", "police", "children", "group_20", "group_100",
                      "flag", "night", "shouting"]
        df['imgpath'] = imgpaths
        df.iloc[:,1:] = np.concatenate(outputs)
        df.sort_values(by = 'imgpath', inplace=True)
        df['protest_close'] = np.abs(df['protest'] - 0.5)
        df_close = df.nsmallest(n,'protest_close')
        img_df_cp = img_df.copy()
        img_df_cp['imgpath'] = img_df_cp.iloc[:,0].apply(lambda x : os.path.join(img_dir,x))
        img_df_cp = img_df_cp[img_df_cp['imgpath'].isin(df_close['imgpath'])]
        return img_df_cp.drop('imgpath',axis=1)    

def calculate_similarities(model, data_loader):
    """Helper function to calculate average pairwise cosine similarity between all images, for each image"""
    feature_extractor = torch.nn.Sequential(
                                    *list(model.children())[:-1]
                                )
    with torch.no_grad():
        outputs = []
        imgpaths = []

        n_imgs = len(data_loader.dataset)
        with tqdm(total=n_imgs) as pbar:
            for i, sample in enumerate(data_loader):
                imgpath, input = sample['imgpath'], sample['image']
                if args.cuda:
                    input = input.cuda()

                input_var = Variable(input)
                output = feature_extractor(input_var)
                outputs.append(output.cpu().data)
                imgpaths += imgpath
                if i < n_imgs / args.batch_size:
                    pbar.update(args.batch_size)
                else:
                    pbar.update(n_imgs%args.batch_size)

        features = torch.cat(outputs, dim = 0)

    cos = lambda m: torch.nn.functional.normalize(m[:,:,0,0]) @ torch.nn.functional.normalize(m[:,:,0,0]).t()
    similarities = cos(features)
    average_sim = torch.mean(similarities, dim = 1)

    return average_sim.numpy()

def eval_one_similarity(img_df, img_dir, model, n, beta=1):
    """Evaluates unlabeled images using similarity density"""

    model.eval()

    dataset = ProtestDataset_AL(img_dir = img_dir, img_df = img_df)
    data_loader = DataLoader(dataset,
                            num_workers = args.workers,
                            batch_size = args.batch_size)
    
    print("Calculating cosine similarities of unlabeled images")
    similarities = calculate_similarities(model, data_loader)

    outputs = []
    imgpaths = []

    print("Calculating model uncertainty of unlabeled images")
    n_imgs = len(data_loader.dataset)
    with tqdm(total=n_imgs) as pbar:
        for i, sample in enumerate(data_loader):
            imgpath, input = sample['imgpath'], sample['image']
            if args.cuda:
                input = input.cuda()

            input_var = Variable(input)
            output = model(input_var)
            outputs.append(output.cpu().data.numpy())
            imgpaths += imgpath
            if i < n_imgs / args.batch_size:
                pbar.update(args.batch_size)
            else:
                pbar.update(n_imgs%args.batch_size)

    df = pd.DataFrame(np.zeros((n_imgs, 14)))
    df.columns = ["img_idx", "imgpath", "protest", "violence", "sign", "photo",
                    "fire", "police", "children", "group_20", "group_100",
                    "flag", "night", "shouting"]
    
    df['imgpath'] = imgpaths
    df.iloc[:,2:] = np.concatenate(outputs)

    """Interested in probabilities closest to 0.5"""
    df['protest_close'] = np.abs(df['protest'] - 0.5)
    
    assert df.shape[0] == len(similarities)

    """Scale all probabilities by similarity scores"""
    # df.iloc[:, 2:] = (1/df.iloc[:, 2:]).mul((similarities ** beta), axis = 0)

    """Scale protest uncertainties by similarity scores"""
    df['protest_close'] = (1 - df['protest_close']) * (similarities ** beta)

    df_close = df.nlargest(n, 'protest_close')

    img_df_cp = img_df.copy()
    img_df_cp['imgpath'] = img_df_cp.iloc[:,0].apply(lambda x : os.path.join(img_dir,x))
    img_df_cp = img_df_cp[img_df_cp['imgpath'].isin(df_close['imgpath'])]
    return img_df_cp.drop('imgpath',axis=1)


def eval_one_data_gradient(img_df, img_dir, model, n):
    """
    return model output of all the images in a directory
    """
    model.eval()
    # make dataloader
    dataset = ProtestDataset_AL(img_dir = img_dir, img_df = img_df)
    data_loader = DataLoader(dataset,
                            num_workers = args.workers,
                            batch_size = args.batch_size)
    # load model

    outputs = []
    imgpaths = []

    n_imgs = len(img_df.iloc[:,0]) #len(os.listdir(img_dir))
    with tqdm(total=n_imgs) as pbar:
        for i, sample in enumerate(data_loader):
            imgpath, input = sample['imgpath'], sample['image']
            if args.cuda:
                input = input.cuda()

            input_var = Variable(input)
            output = model(input_var)
            outputs.append(output.cpu().data.numpy())
            imgpaths += imgpath
            if i < n_imgs / args.batch_size:
                pbar.update(args.batch_size)
            else:
                pbar.update(n_imgs%args.batch_size)


    df = pd.DataFrame(np.zeros((n_imgs, 13)))
    df.columns = ["imgpath", "protest", "violence", "sign", "photo",
                    "fire", "police", "children", "group_20", "group_100",
                    "flag", "night", "shouting"]
    df['imgpath'] = imgpaths
    df.iloc[:,1:] = np.concatenate(outputs)
    df.sort_values(by = 'imgpath', inplace=True)
    df['protest_close'] = np.abs(df['protest'] - 0.5)
    df_close = df.nsmallest(50,'protest_close')
    img_df_cp = img_df.copy()
    img_df_cp['imgpath'] = img_df_cp.iloc[:,0].apply(lambda x : os.path.join(img_dir,x))
    img_df_cp = img_df_cp[img_df_cp['imgpath'].isin(df_close['imgpath'])]

    gradient_df = train_gradient(img_df_cp.drop('imgpath',axis=1), img_dir, model)
    gradient_largest = gradient_df.nlargest(n, 'expected_gradient')

    img_df_cp = img_df_cp[img_df_cp['fname'].isin(gradient_largest['fname'])]

    return img_df_cp.drop('imgpath',axis=1)

def train_gradient(img_df, img_dir, model):
    
    criterion_protest = nn.BCELoss()
    criterion_violence = nn.MSELoss()
    criterion_visattr = nn.BCELoss()
    criterions = [criterion_protest, criterion_violence, criterion_visattr]
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948,  0.4203]])
    
    train_dataset_nl = ProtestDataset(
                        df_imgs= img_df,
                        img_dir = img_dir,
                        transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(
                                    brightness = 0.4,
                                    contrast = 0.4,
                                    saturation = 0.4,
                                    ),
                                transforms.ToTensor(),
                                Lighting(0.1, eigval, eigvec),
                                normalize,
                        ]))
    train_loader_nl = DataLoader(
                train_dataset_nl,
                num_workers = args.workers,
                batch_size = 1,
                shuffle = False
                )

    n_imgs = len(img_df.iloc[:,0])

    label_0s = []
    label_1s = []
    imgs = []
    prob_1s = []
    for i, sample in tqdm(enumerate(train_loader_nl)):
        # measure data loading batch_time
        input, target, img_name = sample['image'], sample['label'], sample['fname']
        #print(target)
        imgs += img_name
        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        input_var = Variable(input)
        output = model(input_var)
        prob_1s.append(output.cpu().data.numpy()[0,0])
        #print(input_var)
        target_var['protest'] = torch.zeros(target_var['protest'].size(), dtype= torch.float64).cuda()

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)

        loss = 0
        for l in losses:
            loss += l
        # back prop
        loss.backward()
        total_norm_0 = 0
        for p in model.parameters():
            if p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                p.grad.data.zero_()
                total_norm_0 += param_norm.item() ** 2
        total_norm_0 = total_norm_0 ** 0.5
        label_0s.append(total_norm_0)


        target_var['protest'] = torch.ones(target_var['protest'].size(), dtype= torch.float64).cuda()

        output = model(input_var)
        losses, scores, N_protest = calculate_loss(output, target_var, criterions)
        
        loss = 0
        for l in losses:
            loss += l
        # back prop
        loss.backward()
        total_norm_1 = 0
        for p in model.parameters():
            if p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                p.grad.data.zero_()
                total_norm_1 += param_norm.item() ** 2
        total_norm_1 = total_norm_1 ** 0.5
        label_1s.append(total_norm_1)

    df = pd.DataFrame(np.zeros((n_imgs, 4)))
    df.columns = ['fname', 'label_0', 'label_1', 'prob_1']
    df['fname'] = imgs
    df['label_0'] = label_0s
    df['label_1'] = label_1s
    df['prob_1'] = prob_1s
    df['expected_gradient'] = (df['label_1'] * df['prob_1']) + (df['label_0'] * (1-df['prob_1']))
    return df 

def eval_loss_decrease(img_df, img_dir, model, n):
    # load predict loss decrease model
    model = new_resnet50()
    model.load_state_dict(torch.load('Mymodel2.pt')) # model directory needs to be set mannually
    model = model.cuda()
    model.eval()

    dataset = ProtestDataset_AL(img_dir=img_dir, img_df=img_df)
    data_loader = DataLoader(dataset,
                             num_workers=args.workers,
                             batch_size=args.batch_size)

    outputs = []
    imgpaths = []

    n_imgs = len(img_df.iloc[:, 0])
    with tqdm(total=n_imgs) as pbar:
        for i, sample in enumerate(data_loader):
            imgpath, input = sample['imgpath'], sample['image']
            if args.cuda:
                input = input.cuda()

            input_var = Variable(input)
            output = model(input_var)
            outputs.append(output.cpu().data.numpy())
            imgpaths += imgpath
            if i < n_imgs / args.batch_size:
                pbar.update(args.batch_size)
            else:
                pbar.update(n_imgs % args.batch_size)

    df = pd.DataFrame(np.zeros((n_imgs, 2)))
    df.columns = ["imgpath", "loss_decrease"]
    df['imgpath'] = imgpaths
    df.iloc[:, 1] = np.concatenate(outputs)
    df.sort_values(by='imgpath', inplace=True)
    #df['protest_close'] = np.abs(df['protest'] - 0.5)
    df_close = df.nsmallest(n, 'loss_decrease')
    img_df_cp = img_df.copy()
    img_df_cp['imgpath'] = img_df_cp.iloc[:, 0].apply(lambda x: os.path.join(img_dir, x))
    img_df_cp = img_df_cp[img_df_cp['imgpath'].isin(df_close['imgpath'])]
    return img_df_cp.drop('imgpath', axis=1)

def adjust_learning_rate_reset(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.5 every epoch for every 5 epochs"""

    lr = args.lr * (0.5 ** (epoch % 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
        
def learning_method(method_id=0,optimizer=None, heuristic_func=None, param1=None, param2=None,param3=None,param4=None,epoch=0):
    
    if method_id == 1:
        # Dont adjust learning rate while AL. Decay it in the end.
        if epoch<100:
            if heuristic_func:
                al_image = heuristic_func(param1,param2,param3,param4)
            else:
                al_image = param1.sample(param4)
            #adjust_learning_rate_dynamic(optimizer, 32611 - len(param1))
        else:
            al_image = None
            adjust_learning_rate(optimizer,epoch-100)

    elif method_id == 2:
        #Train for 20 epochs without any AL. 20-100 do AL and train at a lower learning rate. Do another 50 by decaying without AL
        if epoch<20:
            adjust_learning_rate(optimizer,epoch)
            al_image = None

        elif epoch>=20 and epoch <120:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0002
            if heuristic_func:
                al_image = heuristic_func(param1,param2,param3,param4)
            else:
                al_image = param1.sample(param4)
        else:
            al_image = None
            adjust_learning_rate(optimizer,epoch-120)

    elif method_id == 3:
        #Add and image every 5 epochs. Adjust learning rate and reset lr every 5 epochs
        if epoch <= 500 and epoch % 5 == 0:
            if heuristic_func:
                al_image = heuristic_func(param1,param2,param3,param4)
            else:
                al_image = param1.sample(param4)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        else:
            adjust_learning_rate_reset(optimizer,epoch)
            al_image = None
            
    elif method_id == 4:
        #Train for 20 epochs without any AL. 20-100 do AL and train at a lower learning rate. Do another 50 by decaying without AL
        if epoch<20:
            adjust_learning_rate(optimizer,epoch)
            al_image = None

        elif epoch>=20 and epoch <120:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0002
                args.lr = 0.0002
            if heuristic_func:
                al_image = heuristic_func(param1,param2,param3,param4)
            else:
                al_image = param1.sample(param4)
        else:
            al_image = None
            adjust_learning_rate(optimizer,epoch-120)


    else:
        #Baseline learning method. Add an image every epoch and adjust lr everytime
        adjust_learning_rate(optimizer,epoch)
        if heuristic_func:
            al_image = heuristic_func(param1,param2,param3, param4)
        else:
            al_image = param1.sample(param4)




    return al_image

    



def main():
    global best_loss
    loss_history_train = []
    loss_history_val = []
    data_dir = args.data_dir
    num_label_samples = args.num_label_samples
    n = args.num_samples_added
    img_dir_train = os.path.join(data_dir, "img/train")
    img_dir_val = os.path.join(data_dir, "img/test")
    txt_file_train = os.path.join(data_dir, "annot_train.txt")
    txt_file_val = os.path.join(data_dir, "annot_test.txt")

    # load pretrained resnet50 with a modified last fully connected layer
    model = modified_resnet50()

    # we need three different criterion for training
    criterion_protest = nn.BCELoss()
    criterion_violence = nn.MSELoss()
    criterion_visattr = nn.BCELoss()
    criterions = [criterion_protest, criterion_violence, criterion_visattr]

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU Found")
    if args.cuda:
        model = model.cuda()
        criterions = [criterion.cuda() for criterion in criterions]
    # we are not training the frozen layers
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
                        parameters, args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay
                        )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            loss_history_train = checkpoint['loss_history_train']
            loss_history_val = checkpoint['loss_history_val']
            if args.change_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948,  0.4203]])
    txt_file_train = pd.read_csv(txt_file_train, delimiter="\t").replace('-', 0)
    
    txt_file_train_l = txt_file_train.sample(num_label_samples, random_state=107)
    txt_file_train_nl = txt_file_train.drop(txt_file_train_l.index)

    #`print(len(txt_file_train_l),len(txt_file_train_nl),len(txt_file_train))`
    # train_dataset = ProtestDataset(
    #                     txt_file = txt_file_train,
    #                     img_dir = img_dir_train,
    #                     transform = transforms.Compose([
    #                             transforms.RandomResizedCrop(224),
    #                             transforms.RandomRotation(30),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.ColorJitter(
    #                                 brightness = 0.4,
    #                                 contrast = 0.4,
    #                                 saturation = 0.4,
    #                                 ),
    #                             transforms.ToTensor(),
    #                             Lighting(0.1, eigval, eigvec),
    #                             normalize,
    #                     ]))
    txt_file_val = pd.read_csv(txt_file_val, delimiter="\t").replace('-', 0)
    val_dataset = ProtestDataset(
                    df_imgs = txt_file_val,
                    img_dir = img_dir_val,
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    # train_loader = DataLoader(
    #                 train_dataset,
    #                 num_workers = args.workers,
    #                 batch_size = args.batch_size,
    #                 shuffle = True
    #                 )
    val_loader = DataLoader(
                    val_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size)

    if args.heuristic_id == 1:
        heuristic_func = eval_one_data

    elif args.heuristic_id == 2:
        heuristic_func = eval_one_data_gradient
        pass

    elif args.heuristic_id == 3:
        heuristic_func = eval_one_similarity
        pass
    
    elif args.heuristic_id == 4:
        heuristic_func = eval_loss_decrease
        
    else:
        heuristic_func = None
        
        
    for epoch in range(args.start_epoch, args.epochs):

        train_dataset = ProtestDataset(
                        df_imgs= txt_file_train_l,
                        img_dir = img_dir_train,
                        transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(
                                    brightness = 0.4,
                                    contrast = 0.4,
                                    saturation = 0.4,
                                    ),
                                transforms.ToTensor(),
                                Lighting(0.1, eigval, eigvec),
                                normalize,
                        ]))

        train_loader = DataLoader(
                    train_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size,
                    shuffle = True
                    )

        print(len(txt_file_train_l),len(txt_file_train_nl),len(txt_file_train))

        #adjust_learning_rate(optimizer, epoch)
        loss_history_train_this = train(train_loader, model, criterions,
                                        optimizer, epoch)
        loss_val, loss_history_val_this = validate(val_loader, model,
                                                   criterions, epoch)
        loss_history_train.append(loss_history_train_this)
        loss_history_val.append(loss_history_val_this)

        # loss = loss_val.avg

        is_best = loss_val < best_loss
        if is_best:
            print('best model!!')
        best_loss = min(loss_val, best_loss)


        save_checkpoint({
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict(),
            'best_loss' : best_loss,
            'optimizer' : optimizer.state_dict(),
            'loss_history_train': loss_history_train,
            'loss_history_val': loss_history_val
        }, is_best,filename=f'checkpoint_{args.num_label_samples}_{args.num_samples_added}.pth.tar',
        filename_best_model=f'modelbest_{args.num_label_samples}_{args.num_samples_added}.pth.tar'
        
        )

       

            
        #if heuristic_func:
        al_image = learning_method(args.method_id,optimizer,heuristic_func,txt_file_train_nl,img_dir_train, model, n, epoch)

        # else:
        #     al_image = txt_file_train_nl.sample(1)
        #     adjust_learning_rate(optimizer, epoch)
        #al_image = txt_file_train_nl.sample(1)
        if al_image is not None:
            txt_file_train_l = txt_file_train_l.append(al_image)
            txt_file_train_nl = txt_file_train_nl.drop(al_image.index)
        for param_group in optimizer.param_groups:
            print(f"At epoch {epoch} the lr is : {param_group['lr']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "UCLA-protest",
                        help = "directory path to UCLA-protest",
                        )
    parser.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 8,
                        help = "batch size",
                        )
    parser.add_argument("--epochs",
                        type = int,
                        default = 100,
                        help = "number of epochs",
                        )
    parser.add_argument("--weight_decay",
                        type = float,
                        default = 1e-4,
                        help = "weight decay",
                        )
    parser.add_argument("--lr",
                        type = float,
                        default = 0.01,
                        help = "learning rate",
                        )
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "momentum",
                        )
    parser.add_argument("--print_freq",
                        type = int,
                        default = 10,
                        help = "print frequency",
                        )
    parser.add_argument("--num_label_samples",
                        type = int,
                        default = 100,
                        help = "number of initial labeled samples",
                        )
    
    parser.add_argument("--num_samples_added",
                        type = int,
                        default = 1,
                        help = "number of samples added each epoch",
                        )

    parser.add_argument("--method_id",
                        type = int,
                        default = 0,
                        help = "Which learning rate adjustment to use",
                        )

    parser.add_argument("--heuristic_id",
                        type = int,
                        default = 0,
                        help = "Which heuristic to use for AL",
                        )

    parser.add_argument('--resume',
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--change_lr',
                        action = "store_true",
                        help = "Use this if you want to \
                        change learning rate when resuming")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    args = parser.parse_args()

    if args.cuda:
        protest_idx = protest_idx.cuda()
        violence_idx = violence_idx.cuda()
        visattr_idx = visattr_idx.cuda()


    main()

"""
training heuristic model to predict loss decrease
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image

from util import ProtestDatasetLossTrain,new_resnet50


def train(train_dataloader, model, epochs=10, graph_name='train.png'):
    # Set up hyperparameters first
    criterion = torch.nn.MSELoss()
    lr = 0.01
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    epochs = epochs
    loss_output = [] # record loss change throughout epochs
    pbar = tqdm(range(epochs))
    # Training
    for t in pbar:
        total_loss = 0  # total loss per epoch
        for batch in train_dataloader:
            data, y_true = batch['imgae'],batch['loss']
            data = data.cuda()
            y_true = y_true.cuda()
        # Feed forward to get the logits
            y_pred = model(data)
        # Compute the loss and accuracy
            loss = criterion(y_pred, y_true.unsqueeze(1))
            total_loss += loss.item()
        # zero the gradients before running
        # the backward pass.
            optimizer.zero_grad()
        # Backward pass to compute the gradient
        # of loss w.r.t our learnable params.
            loss.backward()
        # Update params
            optimizer.step()
            pbar.set_postfix({'epoch':t,'loss':'{:.3f}'.format(total_loss)})
        loss_output.append(total_loss)
    # Draw the graph of train
    #print(loss_output)
    plt.plot(np.arange(1, epochs),loss_output[1:])
    plt.xlabel('Epochs')
    plt.ylabel('Training MSE Loss')
    #plt.ylim(0.4, 0.5)
    plt.savefig(graph_name)
    with open('train_loss.txt','w') as f:
        for output in loss_output:
            f.write(str(output)+'\n')


def evaluate(eval_dataloader, model):
    # Set up hyperparameters first
    criterion = torch.nn.MSELoss()
    pbar = tqdm(eval_dataloader)
    total_loss = 0
    for batch in pbar:
        data, y_true = batch['image'],batch['loss']
        y_pred = model(data)
        loss = criterion(y_pred, y_true.unsqueeze(1))
        total_loss += loss.item()
        pbar.set_postfix({'loss':'{:.3f}'.format(loss.item())})
    print('Evaluation result: MSE Loss = ',total_loss/len(eval_dataloader))


def main():
    model = new_resnet50()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948,  0.4203]])

    df = pd.read_csv(args.data_file)
    df.columns = df.columns.values.astype(int)

    
    df_dataset = ProtestDatasetLossTrain(
                    df_imgs = df,
                    img_dir = args.data_dir,
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))

    train_dataset,val_dataset = random_split(df_dataset,[int(0.8*len(df_dataset)),int(0.2*len(df_dataset))])
    """
    train_dataset = ProtestDatasetLossTrain(
                    df_imgs= txt_file_train,
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
    """
    train_loader = DataLoader(
                    train_dataset,
                    batch_size = 50,
                    shuffle = True
                    )
    """            
    val_dataset = ProtestDatasetLossTrain(
                    df_imgs = txt_file_val,
                    img_dir = img_dir_val,
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    """
    val_loader = DataLoader(
                val_dataset,
                batch_size = 100)


    loss_train = train(train_loader, model, args.epochs,args.graph_name)
    loss_val = evaluate(val_loader,model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "UCLA-protest",
                        help = "directory path to UCLA-protest",
                        )
    parser.add_argument("--data_file",
                        type=str,
                        help = "file to store the loss decrease",
                        )          
    parser.add_argument("--epochs",
                        type = int,
                        default = 100,
                        help = "number of epochs",
                        )
    parser.add_argument("--graph_name",
                        type=str,
                        default = "loss_train.png",
                        help = "training loss graph",
                        )


    args = parser.parse_args()

    main()

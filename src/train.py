import torch
import pandas as pd
import numpy as np
import re

import config
from utils import Averager
from utils import get_train_transform
from utils import get_valid_transform
from model import obtain_model
from dataset import WheatDataset


def run(train_path):
    df = pd.read_csv(train_path)
    print(df.shape)
    df['x'] = df['bbox'].apply(lambda x: float(np.array(re.findall("([0-9]+[.]?[0-9]*)", x))[0]))
    df['y'] = df['bbox'].apply(lambda x: float(np.array(re.findall("([0-9]+[.]?[0-9]*)", x))[1]))
    df['w'] = df['bbox'].apply(lambda x: float(np.array(re.findall("([0-9]+[.]?[0-9]*)", x))[2]))
    df['h'] = df['bbox'].apply(lambda x: float(np.array(re.findall("([0-9]+[.]?[0-9]*)", x))[3]))
    df.drop(['bbox'], inplace=True, axis=1)

    # split the data 
    image_ids = df['image_id'].unique()
    valid_ids = image_ids[-665:]
    train_ids = image_ids[-665:]
    train_df = df[df['image_id'].isin(train_ids)]
    valid_df = df[df['image_id'].isin(valid_ids)]
    
    train_dataset = WheatDataset(train_df, config.DIR_TRAIN, get_train_transform())
    valid_dataset = WheatDataset(valid_df, config.DIR_TRAIN, get_valid_transform())

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BS,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BS,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    # Device used is cuda
    device = torch.device('cuda')
    model = obtain_model()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = None

    loss_hist = Averager()
    itr = 1

    for epoch in range(config.EPOCHS):
        loss_hist.reset()

        for images, targets, image_ids in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
            
            itr += 1
    
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        print(f"Epoch #{epoch} loss: {loss_hist.value}")   



if __name__ == "__main__":
    train_path = "../Dataset/train.csv"
    run(train_path)
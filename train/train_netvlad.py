from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np
import yaml
import time
import os
import random
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from tools.database import kitti_dataset
from modules.loss import quadruplet_loss, pose_loss, triplet_loss
from modules.overlapnetvlad import vlad_head
from evaluate import evaluate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
randg = np.random.RandomState()


def train(config):
    root = config["data_root"]["data_root_folder"]
    log_folder = config["training_config"]["log_folder"]
    training_seqs = config["training_config"]["training_seqs"]
    pretrained_vlad_model = config["training_config"]["pretrained_vlad_model"]
    pos_threshold = config["training_config"]["pos_threshold"]
    neg_threshold = config["training_config"]["neg_threshold"]
    batch_size = config["training_config"]["batch_size"]
    epoch = config["training_config"]["epoch"]
    
    log_folder = os.path.join(p, log_folder)
    if (not os.path.exists(log_folder)):
        os.makedirs(log_folder)

    writer = SummaryWriter()
    train_dataset = kitti_dataset(
        root=root,
        seqs=training_seqs,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0)
    vlad = vlad_head().to(device=device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vlad.parameters()),
        lr=1e-5, weight_decay=1e-6)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer,
    #    milestones=[200000, 3200000, 51200000, 1638400000],
    #    gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99,
    )
    loss_function = triplet_loss

    
    if not pretrained_vlad_model == "":
        checkpoint = torch.load(pretrained_vlad_model)
<<<<<<< HEAD
        #model_dict = vlad.state_dict()
        #checkpoint = {k: v for k, v in checkpoint.items() if (k in model_dict and 'mlp' not in k)}
        #model_dict.update(checkpoint)
        #vlad.load_state_dict(model_dict)
        vlad.load_state_dict(checkpoint['state_dict'], strict=False)
=======
        vlad.load_state_dict(checkpoint['state_dict'])
>>>>>>> 6d91ec8a27c3585b24d96da6ba3206fc7cd0ec71

    batch_num = 0
    for i in range(epoch):
        vlad.train()
        for i_batch, sample_batch in tqdm(enumerate(train_loader), total=len(
                train_loader), desc='Train epoch ' + str(i), leave=False):
            optimizer.zero_grad()
            input = torch.cat([sample_batch['query_desc'].flatten(0,1),
                               sample_batch['pos_desc'].flatten(0,1),
                               sample_batch['neg_desc'].flatten(0,1),
                               #sample_batch['other_desc']
                               ], dim=0).to(device)
            out = vlad(input)
            if not out.shape[0] == batch_size * 13:
               continue
            query_fea, pos_fea, neg_fea = torch.split(
                out, [batch_size, batch_size * 2, batch_size * 10], dim=0)
            
            query_fea = query_fea.unsqueeze(1)
            pos_fea = pos_fea.reshape(batch_size, 2, -1)
            neg_fea = neg_fea.reshape(batch_size, 10, -1)
            train_dataset.update_latent_vectors(
                query_fea,
                sample_batch['id'])

            #query_fea1 = query_fea.repeat(1, 2, 1)
            #input_pose1 = torch.cat([query_fea1, pos_fea], dim=2) #1x2x2048
            #query_fea2 = query_fea.repeat(1, 10, 1)
            #input_pose2 = torch.cat([query_fea2, neg_fea], dim=2) #1x10x2048
            #input_pose = torch.cat([input_pose1, input_pose2], dim=1) #1x12x2048
            ##print("input_pose ", input_pose.shape)
            #pre_pose = vlad.pre_dis(input_pose.permute(0, 2, 1))
            ##pre_dis = vlad.pre_dis(torch.hstack([query_fea, pos_fea]).permute(1,0))
            #p_loss = pose_loss(sample_batch['pos_poses'], sample_batch['neg_poses'], pre_pose.permute(0, 2, 1))
            #p_loss *= 1e-3

            #pos_dis, neg_dis, other_dis, loss = loss_function(
            loss = loss_function(query_fea, pos_fea, neg_fea, 0.3)

            print(loss.cpu().item())#,  p_loss.cpu().item())
            #loss = loss + p_loss

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                writer.add_scalar(
                    'loss', loss.cpu().item(), global_step=batch_num)
                #writer.add_scalar(
                #    'p_loss', p_loss.cpu().item(), global_step=batch_num)
                writer.add_scalar(
                    'LR',
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    global_step=batch_num)
                #writer.add_scalar('pos_dis',
                #                  pos_dis.cpu()[0].item(),
                #                  global_step=batch_num)
                #writer.add_scalar('neg_dis',
                #                  neg_dis.cpu()[0].item(),
                #                  global_step=batch_num)
                #writer.add_scalar('other_dis',
                #                  other_dis.cpu()[0].item(),
                #                  global_step=batch_num)

                batch_num += 1

        #recall = evaluate.evaluate_vlad(vlad)
        #recall = 0
        #print("EVAL RECALL:", recall)
            if batch_num % 100 == 10:
                #writer.add_scalar("RECALL",
                #                  recall,
                #                  global_step=batch_num)
                torch.save({'epoch': i,
                        'state_dict': vlad.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'batch_num': batch_num},
                         os.path.join(log_folder, str(batch_num) + ".ckpt"))
                scheduler.step()


if __name__ == '__main__':
    config_file = os.path.join(p, './config/config.yml')
    config = yaml.safe_load(open(config_file))

    train(config)

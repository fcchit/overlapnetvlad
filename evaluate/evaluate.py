import yaml
from tqdm import tqdm
import torch
import sys
import math
import numpy as np
import os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from modules.overlapnetvlad import vlad_head, overlap_head
from tools.utils import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_descriptors(vlad, fea_folder, batch_num):
    fea_files = sorted(os.listdir(fea_folder))
    fea_files = [os.path.join(fea_folder, v) for v in fea_files]
    length = len(fea_files)

    vlad_arr = np.zeros((length, 1024), dtype=np.float32)

    for q_index in tqdm(range(length // batch_num),
                        desc="generating global descriptors",
                        total=length // batch_num):
        batch_files = fea_files[q_index * batch_num:(q_index + 1) * batch_num]
        queries = utils.load_npy_files(batch_files)

        with torch.no_grad():
            input = torch.tensor(queries).float().to(device)
            vlad_out = vlad(input)

        ran = range(q_index * batch_num, (q_index + 1) * batch_num)
        vlad_arr[ran] = vlad_out.detach().cpu().numpy()

    index_edge = length // batch_num * batch_num
    if index_edge < length:
        batch_files = fea_files[index_edge:length]
        queries = utils.load_npy_files(batch_files)

        with torch.no_grad():
            input = torch.tensor(queries).float().to(device)
            vlad_out = vlad(input)

        ran = range(index_edge, length)
        vlad_arr[ran] = vlad_out.detach().cpu().numpy()

    return vlad_arr


def evaluate_vlad(vlad, seq, topk=1):
    root = "/home/fuchencan/datasets/KITTI/datasets/sequences"
    
    config_file = os.path.join(p, './config/config.yml')
    config = yaml.safe_load(open(config_file))

    #seq = config["test_config"]["seq"]
    #root = config["data_root"]["data_root_folder"]
    
    th_min = config["test_config"]["th_min"]
    th_max = config["test_config"]["th_max"]
    th_max_pre = config["test_config"]["th_max_pre"]
    skip = config["test_config"]["skip"]
    batch_num = config["test_config"]["batch_num"]

    vlad.eval()
    fea_folder = os.path.join(root, seq, "BEV_FEA_TRN_B1")
    vlad_arr = generate_descriptors(vlad, fea_folder, batch_num)

    pose = np.genfromtxt(os.path.join(root.replace(
        'sequences', 'poses_semantic'), seq + '.txt'))[:, [3, 11]]
    length = len(pose)

    #topk = max(25, int(length * 1e-2))
    print(f"evaluate: {seq}, topk: {topk}, len:{length}")

    correct_at_k = np.zeros(topk)
    whole_test_size = 0
    for i in tqdm(range(length), desc="evaluating", total=length):
        pos_dis = np.linalg.norm(pose - pose[i], axis=1)

        pos_dis[max(i - skip, 0):] = np.inf
        mask = (pos_dis < th_min)
        pos_dis[mask] = np.inf

        mindis_gt = np.min(pos_dis)
        if mindis_gt < th_max:
            whole_test_size += 1

            vlad_dis = np.linalg.norm(vlad_arr - vlad_arr[i], axis=1)

            vlad_dis[max(i - skip, 0):] = np.inf
            vlad_dis[mask] = np.inf

            vlad_topks = np.argsort(vlad_dis)[:topk]

            for k, k_idx in enumerate(vlad_topks):
                dis_gt = pos_dis[k_idx]
                if dis_gt < th_max_pre:
                    correct_at_k[k:] += 1
                    break
    recalls = correct_at_k / whole_test_size
    vlad.train()
    return recalls


def evaluate_overlapnetvlad(vlad_arr, overlapnetvlad, seq, topk=25, topn=1):
    config_file = os.path.join(p, './config/config.yml')
    config = yaml.safe_load(open(config_file))

    #seq = config["test_config"]["seq"]
    root = "/home/fuchencan/datasets/KITTI/datasets/sequences"
    #root = config["data_root"]["data_root_folder"]
    th_min = config["test_config"]["th_min"]
    th_max = config["test_config"]["th_max"]
    th_max_pre = config["test_config"]["th_max_pre"]
    skip = config["test_config"]["skip"]
    batch_num = config["test_config"]["batch_num"]

    
    # vlad.eval()
    # fea_folder = os.path.join(root, seq, "BEV_FEA_TRN_B1")
    # vlad_arr = generate_descriptors(vlad, fea_folder, batch_num)

    feature_files = sorted(os.listdir(fea_folder))
    feature_files = [os.path.join(fea_folder, v) for v in feature_files]

    pose = np.genfromtxt(os.path.join(root.replace(
        'sequences', 'poses_semantic'), seq + '.txt'))[:, [3, 11]]
    length = len(pose)
    
    # topn = max(25, int(length*1e-2))
    #topk = int(length*1e-2)
    print(f"evaluate: {seq}, topk: {topk}, topn: {topn}, len:{length}")

    correct_at_n = np.zeros(topn)
    whole_test_size = 0
    for i in tqdm(range(length), desc="evaluating", total=length):
        pos_dis = np.linalg.norm(pose - pose[i], axis=1)

        pos_dis[max(i - skip, 0):] = np.inf
        mask = (pos_dis < th_min)
        pos_dis[mask] = np.inf

        mindis_gt = np.min(pos_dis)
        if mindis_gt < th_max:
            whole_test_size += 1

            vlad_dis = np.linalg.norm(vlad_arr - vlad_arr[i], axis=1)

            vlad_dis[max(i - skip, 0):] = np.inf
            vlad_dis[mask] = np.inf

            vlad_topks = np.argsort(vlad_dis)[:topk]

            overlap_scores = np.zeros(length, dtype='float32')

            feai = utils.load_npy_files([feature_files[i]])
            feai = torch.from_numpy(feai).to(device)
            for k in vlad_topks:
                feaj = utils.load_npy_files([feature_files[k]])

                with torch.no_grad():
                    feaj = torch.from_numpy(feaj).to(device)
                    overlap, _ = overlapnetvlad(
                        torch.cat([feai, feaj]).permute(0, 2, 3, 1))

                overlap_scores[k] = overlap.detach().cpu().numpy()

            overlap_topns = np.argsort(-overlap_scores)[:topn]
            for n, n_idx in enumerate(overlap_topns):
                dis_gt = pos_dis[n_idx]
                if dis_gt < th_max_pre:
                    correct_at_n[n:] += 1
                    break
    recalls = correct_at_n / whole_test_size
    vlad.train()
    return recalls


if __name__ == '__main__':
    vlad = vlad_head().to(device)
    model_file = "/home/fuchencan/OverlapNetVLAD/log/fine/08291041.ckpt"
    checkpoint = torch.load(model_file)
    vlad.load_state_dict(checkpoint['state_dict'], strict=False)

    print("checkpoint ", model_file)
    
    #seqs = ["07", "02", "05", "00", "06", "08"]
    #for seq in seqs:
    #    print(f"seq: {seq}, recall@N\n", evaluate_vlad(vlad, seq, topk=5))

    overlap = overlap_head(32).to(device)
    checkpoint = torch.load(os.path.join(p, "./models/overlap.ckpt"))
    overlap.load_state_dict(checkpoint['state_dict'])
    
    seqs = ["00", "02", "05", "06", "07", "08"]
    kitti_lengths = [4541, 4661, 2761, 1101, 1101, 4071]
    
    config_file = os.path.join(p, './config/config.yml')
    config = yaml.safe_load(open(config_file))
    root = "/home/fuchencan/datasets/KITTI/datasets/sequences"
    batch_num = config["test_config"]["batch_num"]

    for s, seq in enumerate(seqs):
    #    print(f"seq: {seq}, recall@N\n", evaluate_vlad(vlad, seq, topk=5))
        
        vlad.eval()
        fea_folder = os.path.join(root, seq, "BEV_FEA_TRN_B1")
        vlad_arr = generate_descriptors(vlad, fea_folder, batch_num)

        topks = [10, 15, 20, 25, int(kitti_lengths[s] * 1e-2)]
        for topk in topks:
            recalls = evaluate_overlapnetvlad(vlad_arr, overlap, seq, topk=topk, topn=1)

            for i in range(len(recalls)):
                print(i+1, recalls[i])


import yaml
from tqdm import tqdm
import torch
import sys
import numpy as np
import os

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from modules.overlapnetvlad import vlad_head, overlap_head
from tools.utils import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kitti_lengths = {"00": 4541, "02": 4661, "05": 2761, "06": 1101, "07": 1101, "08": 4071}


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_descriptors(vlad, fea_folder, batch_num):
    fea_files = sorted(os.listdir(fea_folder))
    fea_files = [os.path.join(fea_folder, v) for v in fea_files]
    length = len(fea_files)
    vlad_arr = np.zeros((length, 1024), dtype=np.float32)

    for q_index in tqdm(range((length + batch_num - 1) // batch_num), desc="Generating global descriptors"):
        batch_files = fea_files[q_index * batch_num:min((q_index + 1) * batch_num, length)]
        queries = utils.load_npy_files(batch_files)

        with torch.no_grad():
            input_tensor = torch.tensor(queries).float().to(device)
            vlad_out = vlad(input_tensor)

        vlad_arr[q_index * batch_num:min((q_index + 1) * batch_num, length)] = vlad_out.detach().cpu().numpy()

    return vlad_arr


def evaluate_vlad(root, vlad_arr, seq, topk, th_min, th_max, th_max_pre, skip):
    pose = np.genfromtxt(os.path.join(root, seq, "poses.txt"))[:, [3, 11]]
    length = len(pose)
    correct_at_k = np.zeros(topk)
    whole_test_size = 0

    for i in tqdm(range(length), desc="Evaluating", total=length):
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
    
    recall = correct_at_k / whole_test_size if whole_test_size > 0 else np.zeros(topk)
    print(f"Coarse Recall@{topk}: {recall}")
    
    return recall


def evaluate_overlapnetvlad(root, vlad_arr, overlapnetvlad, seq, topk, topn, th_min, th_max, th_max_pre, skip):
    feature_files = sorted(os.listdir(os.path.join(root, seq, "BEV_FEA")))
    feature_files = [os.path.join(root, seq, "BEV_FEA", v) for v in feature_files]
    pose = np.genfromtxt(os.path.join(root, seq, "poses.txt"))[:, [3, 11]]
    length = len(pose)
    correct_at_n = np.zeros(topn)
    whole_test_size = 0

    for i in tqdm(range(length), desc="Evaluating", total=length):
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
            overlap_scores = np.zeros(length, dtype="float32")

            feai = utils.load_npy_files([feature_files[i]])
            feai = torch.from_numpy(feai).to(device)
            for k in vlad_topks:
                feaj = utils.load_npy_files([feature_files[k]])
                with torch.no_grad():
                    feaj = torch.from_numpy(feaj).to(device)
                    overlap, _ = overlapnetvlad(torch.cat([feai, feaj]).permute(0, 2, 3, 1))
                overlap_scores[k] = overlap.detach().cpu().numpy()

            overlap_topns = np.argsort(-overlap_scores)[:topn]
            for n, n_idx in enumerate(overlap_topns):
                dis_gt = pos_dis[n_idx]
                if dis_gt < th_max_pre:
                    correct_at_n[n:] += 1
                    break

    recall = correct_at_n / whole_test_size if whole_test_size > 0 else np.zeros(topn)
    print(f"Coarse-to-Fine Recall@{topk}: {recall}")

    return recall


if __name__ == "__main__":
    config = load_config(os.path.join(p, "./config/config.yml"))

    vlad = vlad_head().to(device)
    model_file = config["training_config"]["pretrained_vlad_model"]
    checkpoint = torch.load(model_file)
    vlad.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint loaded from", model_file)

    overlap = overlap_head(32).to(device)
    checkpoint = torch.load(os.path.join(p, "./models/overlap.ckpt"))
    overlap.load_state_dict(checkpoint["state_dict"])

    seqs = config["evaluate_config"]["seqs"]
    root = config["data_root"]["data_root_folder"]
    batch_num = config["evaluate_config"]["batch_num"]

    for seq in seqs:
        print(f"Processing Seq {seq}")

        vlad.eval()
        fea_folder = os.path.join(root, seq, "BEV_FEA")
        vlad_arr = generate_descriptors(vlad, fea_folder, batch_num)

        topks = [1, 10, 15, 20, 25, int(kitti_lengths[seq] * 1e-2)]
        for topk in topks:
            print(f"Evaluate Recall@{topk} of the coarse matching stage")
            recall = evaluate_vlad(root, vlad_arr, seq, topk=topk,
                                               th_min=config["evaluate_config"]["th_min"],
                                               th_max=config["evaluate_config"]["th_max"],
                                               th_max_pre=config["evaluate_config"]["th_max_pre"],
                                               skip=config["evaluate_config"]["skip"])

            print(f"Evaluate Recall@{topk} of the whole coarse-to-fine approach")
            recall = evaluate_overlapnetvlad(root, vlad_arr, overlap, seq, topk=topk, topn=1,
                                               th_min=config["evaluate_config"]["th_min"],
                                               th_max=config["evaluate_config"]["th_max"],
                                               th_max_pre=config["evaluate_config"]["th_max_pre"],
                                               skip=config["evaluate_config"]["skip"])
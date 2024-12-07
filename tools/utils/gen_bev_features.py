import yaml
from tqdm import tqdm
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from modules.net import Backbone
from tools.utils import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_kitti(net, scan_folder, dst_folder, batch_num=1):
    files = sorted(os.listdir(scan_folder))
    files = [os.path.join(scan_folder, v) for v in files]
    length = len(files)

    net.train()
    for q_index in tqdm(range(length // batch_num), total=length // batch_num):
        batch_files = files[q_index * batch_num:(q_index + 1) * batch_num]
        with torch.no_grad():
            queries = utils.load_pc_files(batch_files).to(device)
            fea_out = net(queries).cpu().numpy()

        for i in range(len(batch_files)):
            fea_file = os.path.join(dst_folder, os.path.basename(batch_files[i]).replace('.bin', '.npy'))
            np.save(fea_file, fea_out[i])

    index_edge = length // batch_num * batch_num
    if index_edge < length:
        batch_files = files[index_edge:length]
        with torch.no_grad():
            queries = utils.load_pc_files(batch_files).to(device)
            fea_out = net(queries).cpu().numpy()

        for i in range(len(batch_files)):
            fea_file = os.path.join(dst_folder, os.path.basename(batch_files[i]).replace('.bin', '.npy'))
            np.save(fea_file, fea_out[i])


if __name__ == "__main__":
    config = yaml.safe_load(open('./config/config.yml'))

    root = config["data_root"]["data_root_folder"]
    ckpt = config["extractor_config"]["pretrained_backbone_model"]
    seqs = config["extractor_config"]["seqs"]
    batch_num = config["extractor_config"]["batch_num"]

    net = Backbone(32).to(device)
    checkpoint = torch.load(ckpt)
    net.load_state_dict(checkpoint['state_dict'])

    for seq in seqs:
        print(f"Extracting BEV features of Seq {seq}")
        scan_folder = os.path.join(root, seq, "velodyne")
        dst_folder = os.path.join(root, seq, "BEV_FEA")

        os.makedirs(dst_folder, exist_ok=True)

        extract_kitti(net, scan_folder, dst_folder, batch_num)

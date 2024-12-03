from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from scipy.linalg import norm
from tqdm import tqdm
import threading

import sys
p = os.path.dirname((os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)
from net import BEVNet


def load_voxel(data,
               coords_range_xyz=[-50., -50, -4, 50, 50, 3],
               div_n=[256, 256, 32]):
        div = [(coords_range_xyz[3] - coords_range_xyz[0]) / div_n[0],
            (coords_range_xyz[4] - coords_range_xyz[1]) / div_n[1],
            (coords_range_xyz[5] - coords_range_xyz[2]) / div_n[2]]
        id_x = (data[:, 0] - coords_range_xyz[0]) / div[0]
        id_y = (data[:, 1] - coords_range_xyz[1]) / div[1]
        id_z = (data[:, 2] - coords_range_xyz[2]) / div[2]
        all_id = torch.cat(
            [id_x.reshape(-1, 1), id_y.reshape(-1, 1), id_z.reshape(-1, 1)], axis=1).long()

        mask = (all_id[:, 0] >= 0) & (all_id[:, 1] >= 0) & (all_id[:, 2] >= 0) & (
            all_id[:, 0] < div_n[0]) & (all_id[:, 1] < div_n[1]) & (all_id[:, 2] < div_n[2])
        all_id = all_id[mask]
        data = data[mask]
        ids, _, _ = torch.unique(
            all_id, return_inverse=True, return_counts=True, dim=0)

        return ids


def load_pc_file(filename,
                 coords_range_xyz=[-50., -50, -4, 50, 50, 3],
                 div_n=[256, 256, 32],
                 is_pcd=False):
        if is_pcd:
            pc = load_pcd(filename)
        else:
            pc = np.fromfile(filename, dtype="float32").reshape(-1, 4)[:, :3]

        pc = torch.from_numpy(pc)
        ids = load_voxel(pc,
                        coords_range_xyz=coords_range_xyz,
                        div_n=div_n)
        voxel_out = torch.zeros(div_n)
        voxel_out[ids[:, 0], ids[:, 1], ids[:, 2]] = 1
        return voxel_out

def rot3d(axis, angle):
        ei = np.ones(3, dtype='bool')
        ei[axis] = 0
        i = np.nonzero(ei)[0]
        m = np.eye(4)
        c, s = np.cos(angle), np.sin(angle)
        m[i[0], i[0]] = c
        m[i[0], i[1]] = -s
        m[i[1], i[0]] = s
        m[i[1], i[1]] = c
        return m

def occ_pcd(points, state_st=6, max_range=np.pi):
            rand_state = random.randint(state_st, 10)
            if rand_state > 9:
                rand_start = random.uniform(-np.pi, np.pi)
                rand_end = random.uniform(rand_start, min(np.pi, rand_start + max_range))
                angles = np.arctan2(points[:, 1], points[:, 0])
                return points[(angles < rand_start) | (angles > rand_end)]
            else:
                return points

def apply_transform(pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

class kitti_dataset(Dataset):
    def __init__(self, root, seqs, pos_threshold, neg_threshold) -> None:
        super().__init__()
        self.root = root
        self.seqs = seqs
        self.poses = []
        for seq in seqs:
            pose = np.genfromtxt(os.path.join(root, seq, 'poses.txt'))[:, [3, 11]]
            self.poses.append(pose)
        self.pairs = {}
        self.randg = np.random.RandomState()

        key = 0
        acc_num = 0
        for i in range(len(self.poses)):
            pose = self.poses[i]
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose**2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))
            id_pos = np.argwhere((dis < pos_threshold) & (dis > 0))
            id_neg = np.argwhere(dis < neg_threshold)
            for j in range(len(pose)):
                positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
                negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
                self.pairs[key] = {
                    "query_seq": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": set(
                        negatives.tolist())}
                key += 1
            acc_num += len(pose)
        self.all_ids = set(range(len(self.pairs)))
        self.traing_latent_vectors = torch.zeros((len(self.pairs), 1024)).cuda()

        self.net = BEVNet(32).cuda()
        checkpoint = torch.load("/home/fuchencan/OverlapNetVLAD/models/bevnet.ckpt")
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.train()

    def get_random_positive(self, idx, num):
        positives = self.pairs[idx]["positives"]
        randid = np.random.randint(0, len(positives) - 1, num).tolist()
        
        return [positives[idx] for idx in randid]

    def get_random_negative(self, idx, num):
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        randid = np.random.randint(0, len(negatives) - 1, num).tolist()

        return [negatives[idx] for idx in randid]

    def get_random_hard_positive(self, idx, num):
        qurey_vec = self.traing_latent_vectors[idx]
        if qurey_vec.sum() == 0:
            return self.get_random_positive(idx, num) 

        random_pos = self.pairs[idx]["positives"]
        #latent_vecs = []
        #for j in range(len(random_pos)):
        #    latent_vecs.append(self.traing_latent_vectors[random_pos[j]])
        #print(type(latent_vecs[0]))
        #latent_vecs = np.array(latent_vecs)
        #query_vec = self.traing_latent_vectors[idx]
        #query_vec = query_vec.reshape(1, -1)
        #query_vec = np.repeat(query_vec, latent_vecs.shape[0], axis=0)
        #diff = query_vec - latent_vecs
        #diff = np.linalg.norm(diff, axis=1)
        #maxid = np.argmax(diff)
        random_pos = torch.Tensor(random_pos).long().cuda() 
        latent_vecs = self.traing_latent_vectors[random_pos]
        mask = latent_vecs.sum(dim=1) != 0
        latent_vecs = latent_vecs[mask]
        query_vec = self.traing_latent_vectors[idx]
        query_vec = query_vec.reshape(1, -1)
        diff = query_vec - latent_vecs
        diff = torch.linalg.norm(diff, dim=1)
        maxid = torch.argsort(diff)[-num:]

        return random_pos[maxid].tolist()

    def get_random_hard_negative(self, idx, num):
        qurey_vec = self.traing_latent_vectors[idx]
        #if qurey_vec is None:
        #    randid = random.randint(0, len(random_neg) - 1)
        #    return random_neg[randid]
        if qurey_vec.sum() == 0:
            return self.get_random_negative(idx, num)

        random_neg = list(self.all_ids - self.pairs[idx]["negatives"])
        #latent_vecs = []
        #for j in range(len(random_neg)):
        #    latent_vecs.append(self.traing_latent_vectors[random_neg[j]])

        #latent_vecs = np.array(latent_vecs)
        #query_vec = self.traing_latent_vectors[idx]
        #query_vec = query_vec.reshape(1, -1)
        #query_vec = np.repeat(query_vec, latent_vecs.shape[0], axis=0)
        #diff = query_vec - latent_vecs
        #diff = np.linalg.norm(diff, axis=1)
        #minid = np.argmin(diff)
        random_neg = torch.Tensor(random_neg).long().cuda()
        latent_vecs = self.traing_latent_vectors[random_neg]
        mask = latent_vecs.sum(dim=1) != 0
        latent_vecs = latent_vecs[mask]

        query_vec = self.traing_latent_vectors[idx]
        query_vec = query_vec.reshape(1, -1)
        diff = query_vec - latent_vecs
        diff = torch.linalg.norm(diff, dim=1)
        minid = torch.argsort(diff)[:num]
        return random_neg[minid].tolist()

    def get_other_neg(self, id_pos, id_neg):
        random_neg = list(
            self.all_ids -
            self.pairs[id_pos]["negatives"] -
            self.pairs[id_neg]["negatives"])
        randid = random.randint(0, len(random_neg) - 1)

        return random_neg[randid]

    def gre_fea_and_save(self, idx, save_path):
        query = self.pairs[idx]
        seq = self.seqs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        file = os.path.join(self.root, seq, 'velodyne', id + '.bin')
        query = load_pc_file(file).cuda().unsqueeze(0)
        query = torch.cat([query, query])
        fea_out = self.net.extract_feature(query).detach().cpu().numpy()
        np.save(save_path, fea_out)

    def update_latent_vectors(self, fea, idx):
        for i in range(len(idx)):
            self.traing_latent_vectors[idx[i]] = fea[i]

    def load_fea(self, idx):
        query = self.pairs[idx]
        seq = self.seqs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        file = os.path.join(self.root, seq, "velodyne", id + '.bin')
        
        pc = np.fromfile(file, dtype='float32').reshape(-1, 4)[:, 0:3]
        T = rot3d(2, 2. * self.randg.rand(1) * np.pi)
        pc = apply_transform(pc, T)
        pc = occ_pcd(pc, state_st=6, max_range=np.pi)
        pc = torch.from_numpy(pc).cuda()
        
        ids = load_voxel(pc)
        voxel_out = torch.zeros([256, 256, 32])
        voxel_out[ids[:, 0], ids[:, 1], ids[:, 2]] = 1
        query = voxel_out.cuda().unsqueeze(0)

        fea = self.net.extract_feature(query)#.detach().cpu().numpy()
        return fea 

    def load_pose(self, idx):
        query = self.pairs[idx]
        seq = query["query_seq"]
        query_id = query["query_id"]
        pose = self.poses[seq][query_id]

        return pose
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pos_num = 2
        neg_num = 10

        queryid = idx % len(self.pairs)
        posid = self.get_random_hard_positive(queryid, pos_num)
        negid = self.get_random_hard_negative(queryid, neg_num)
        #otherid = self.get_other_neg(queryid, negid)
        
        #query_pose = torch.Tensor(self.load_pose(queryid)).cuda()
        #pos_poses = torch.zeros((pos_num, 2)).cuda()
        #for i in range(pos_num):
        #    p = self.load_pose(posid[i])
        #    pos_poses[i] = torch.Tensor(p)
        ##print(query_pose, '\n', pos_poses)
        #pos_poses -= query_pose
        
        #neg_poses = torch.zeros((neg_num, 2)).cuda()
        #for i in range(neg_num):
        #    p = self.load_pose(negid[i])
        #    neg_poses[i] = torch.Tensor(p)
        ##print(query_pose, '\n', pos_poses)
        #neg_poses -= query_pose
        ##print("pose shape ", pos_poses.shape, neg_poses.shape)
        ##pos_dis = np.linalg.norm(query_pose - pos_pose)

        query_fea = self.load_fea(queryid)
        query_fea.unsqueeze(0)
        pos_feas = torch.zeros((pos_num, 512, 32, 32)).cuda()
        for i in range(pos_num):
            pos_feas[i] = self.load_fea(posid[i])
        neg_feas = torch.zeros((neg_num, 512, 32, 32)).cuda()
        for i in range(neg_num):
            neg_feas[i] = self.load_fea(negid[i])
        #other_fea = self.load_fea(otherid)

        return {
            "id": queryid,
            "query_desc": query_fea,
            "pos_desc": pos_feas,
            "neg_desc": neg_feas,
            #"other_desc": other_voxel,
            #"pos_dis": pos_dis
        #    "pos_poses" : pos_poses,
        #    "neg_poses" : neg_poses,
            }


if __name__ == "__main__":
    dataset = kitti_dataset(
        root="/home/fuchencan/datasets/KITTI/datasets/sequences",
        seqs=["99"],
        pos_threshold=10,
        neg_threshold=50)
    for i in range(len(dataset)):
        d = dataset[random.randint(0, len(dataset) - 1)]
        break

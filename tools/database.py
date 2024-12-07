from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random


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
                    "negatives": set(negatives.tolist())}
                key += 1
            acc_num += len(pose)
        self.all_ids = set(range(len(self.pairs)))
        self.traing_latent_vectors = torch.zeros((len(self.pairs), 1024)).cuda()

    def get_random_positive(self, idx, num):
        positives = self.pairs[idx]["positives"]
        randid = np.random.randint(0, len(positives), num).tolist()
        return [positives[i] for i in randid]

    def get_random_negative(self, idx, num):
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        randid = np.random.randint(0, len(negatives), num).tolist()
        return [negatives[i] for i in randid]

    def get_random_hard_positive(self, idx, num):
        query_vec = self.traing_latent_vectors[idx]
        if query_vec.sum() == 0:
            return self.get_random_positive(idx, num)

        random_pos = self.pairs[idx]["positives"]
        random_pos = torch.Tensor(random_pos).long().cuda()
        latent_vecs = self.traing_latent_vectors[random_pos]
        mask = latent_vecs.sum(dim=1) != 0
        latent_vecs = latent_vecs[mask]
        query_vec = self.traing_latent_vectors[idx].unsqueeze(0)
        diff = query_vec - latent_vecs
        diff = torch.linalg.norm(diff, dim=1)
        maxid = torch.argsort(diff)[-num:]
        return random_pos[maxid].tolist()

    def get_random_hard_negative(self, idx, num):
        query_vec = self.traing_latent_vectors[idx]
        if query_vec.sum() == 0:
            return self.get_random_negative(idx, num)

        random_neg = list(self.all_ids - self.pairs[idx]["negatives"])
        random_neg = torch.Tensor(random_neg).long().cuda()
        latent_vecs = self.traing_latent_vectors[random_neg]
        mask = latent_vecs.sum(dim=1) != 0
        latent_vecs = latent_vecs[mask]
        query_vec = self.traing_latent_vectors[idx].unsqueeze(0)
        diff = query_vec - latent_vecs
        diff = torch.linalg.norm(diff, dim=1)
        minid = torch.argsort(diff)[:num]
        return random_neg[minid].tolist()

    def get_other_neg(self, id_pos, id_neg):
        random_neg = list(self.all_ids - self.pairs[id_pos]["negatives"] - self.pairs[id_neg]["negatives"])
        randid = random.randint(0, len(random_neg) - 1)
        return random_neg[randid]

    def update_latent_vectors(self, fea, idx):
        for i in range(len(idx)):
            self.traing_latent_vectors[idx[i]] = fea[i]

    def load_fea(self, idx):
        query = self.pairs[idx]
        seq = self.seqs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        file = os.path.join(self.root, seq, "BEV_FEA", id + '.npy')
        fea = np.load(file)
        fea = torch.from_numpy(fea).cuda()
        return fea

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

        query_fea = self.load_fea(queryid).unsqueeze(0)

        pos_feas = torch.zeros((pos_num, 512, 32, 32)).cuda()
        for i in range(pos_num):
            pos_feas[i] = self.load_fea(posid[i])

        neg_feas = torch.zeros((neg_num, 512, 32, 32)).cuda()
        for i in range(neg_num):
            neg_feas[i] = self.load_fea(negid[i])

        return {
                "id": queryid,
                "query_desc": query_fea,
                "pos_desc": pos_feas,
                "neg_desc": neg_feas,
               }


if __name__ == "__main__":
    dataset = kitti_dataset(
        root="/home/fuchencan/datasets/KITTI/datasets/sequences",
        seqs=["99"],
        pos_threshold=10,
        neg_threshold=50
    )
    for i in range(len(dataset)):
        d = dataset[random.randint(0, len(dataset) - 1)]
        break

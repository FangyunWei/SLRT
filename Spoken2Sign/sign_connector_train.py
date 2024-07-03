import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle, os
import numpy as np
from tqdm import tqdm
import utils
import queue


class MyDataset(Dataset):
    def __init__(self, kps_path, pairs_path, joint_idx):
        with open(kps_path, 'rb') as f:
            self.kps = pickle.load(f)
        with open(pairs_path, 'rb') as f:
            self.pairs = pickle.load(f)
        self.joint_idx = joint_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clip1, clip2 = self.pairs[idx]
        end, start = clip1['end'], clip2['start']
        label = torch.tensor(start - end).long()
        kps1 = torch.tensor(self.kps[clip1['video_file']]['keypoints_3d'][end][self.joint_idx]).float()
        kps2 = torch.tensor(self.kps[clip2['video_file']]['keypoints_3d'][start][self.joint_idx]).float()
        return {'kps_prev': kps1, 'kps_next': kps2, 'labels': label}

    def collate_fn(self, batch):
        kps_prev = [s['kps_prev'] for s in batch]
        kps_next = [s['kps_next'] for s in batch]
        label = [s['label'] for s in batch]
        return {'kps_prev': torch.stack(kps_prev, dim=0), 
                'kps_next': torch.stack(kps_next, dim=0),
                'labels': torch.stack(label, dim=0)}

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        # x = torch.relu(self.fc4(x))
        return x

if __name__ == '__main__':
    utils.set_seed(0)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    joint_idx = np.array([3, 4, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66], dtype=np.int32)
    lr = 2e-4
    num_epoch = 400
    batch_size = 128  

    root_dir = '../../data/phoenix'
    train_dataset = MyDataset(kps_path=os.path.join(root_dir, 'keypoints_3d_mesh.pkl'),  #os.path.join(root_dir, 'keypoints_projected_3D.pkl'),
                              pairs_path=os.path.join(root_dir, 'iso_clip_pairs.train'),
                              joint_idx=joint_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MyDataset(kps_path=os.path.join(root_dir, 'keypoints_3d_mesh.pkl'),  #os.path.join(root_dir, 'keypoints_projected_3D.pkl'),
                              pairs_path=os.path.join(root_dir, 'iso_clip_pairs.dev'),
                              joint_idx=joint_idx)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=len(joint_idx)*3*2+len(joint_idx))
    # model = MLP(input_dim=len(joint_idx)*3*2)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=0, T_max=num_epoch,)

    criterion = nn.L1Loss()
    best_val_loss = float('inf')
    best_epoch = 0
    ckpt_queue = queue.Queue(maxsize=1)
    for epoch in tqdm(range(num_epoch)):
        train_loss = 0.0
        val_loss = 0.0

        # train
        model.train()
        for i, batch in enumerate(train_loader):
            #[B,N,3], [B,N,3], [B]
            kps_prev, kps_next, labels = batch['kps_prev'].to(device), batch['kps_next'].to(device), batch['labels'].to(device)
            kps_dis = torch.sqrt(((kps_prev-kps_next)**2).sum(dim=-1))  #[B,N]
            inputs = torch.cat([kps_prev.flatten(start_dim=1), kps_next.flatten(start_dim=1), kps_dis], dim=1)  #[B,N*6+1]
            # inputs = torch.cat([kps_prev.flatten(start_dim=1), kps_next.flatten(start_dim=1)], dim=1)  #[B,N*6+1]
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_loss /= len(train_loader)
        print('epoch:', epoch, 'train_loss:', train_loss)

        # eval
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                kps_prev, kps_next, labels = batch['kps_prev'].to(device), batch['kps_next'].to(device), batch['labels'].to(device)
                kps_dis = torch.sqrt(((kps_prev-kps_next)**2).sum(dim=-1))  #[B,N]
                inputs = torch.cat([kps_prev.flatten(start_dim=1), kps_next.flatten(start_dim=1), kps_dis], dim=1)  #[B,N*6+1]
                # inputs = torch.cat([kps_prev.flatten(start_dim=1), kps_next.flatten(start_dim=1)], dim=1)  #[B,N*6+1]
                labels = labels.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                fname = './data/connector_ep{:03d}.pth'.format(epoch)
                torch.save(model.state_dict(), fname)
                print('save')
                if ckpt_queue.full():
                    to_del = ckpt_queue.get()
                    try:
                        os.remove(to_del)
                    except:
                        print(to_del, 'already removed')
                ckpt_queue.put(fname)
            print('best val:', best_val_loss, 'best epoch:', best_epoch)

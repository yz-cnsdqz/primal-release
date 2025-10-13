import torch
import os
from os.path import join
from torch.utils.data import Dataset
import numpy as np
import glob
from ..utils.pylogger import get_pylogger


from collections import Counter

logger = get_pylogger(__name__)


class AMASS_SMPLX_NEUTRAL(Dataset):
    def __init__(self, cfg):
        super().__init__()
        datapath = cfg["path"]
        self.seq_len = seq_len = cfg["seq_len"]
        subsets = cfg["subsets"]
        self.read_all_files_to_ram = cfg["read_all_files_to_ram"]
        self.tgt_fps = cfg["framerate"]
        self.shape_noise_sigma = cfg.get("shape_noise_sigma", -1)
        self.reverse_time_dimension = cfg.get("reverse_time_dimension", False)

        self.npz_files = []
        for subset in subsets:
            self.npz_files += sorted(glob.glob(join(datapath,subset, "*/*.npz"),recursive=True))
            
        if self.read_all_files_to_ram:
            self.all_seq = []
            for file in self.npz_files:
                with open(file, 'rb') as f:
                    data = np.load(f, allow_pickle=True)
                    data_dict = {key: data[key] for key in data}
                    self.all_seq.append(data_dict)
        else:
            self.all_seq = self.npz_files


    def __len__(self):
        return len(self.all_seq)

    def __getitem__(self, index):
        if self.read_all_files_to_ram:
            data = self.all_seq[index] 
        else:
            with open(self.all_seq[index], 'rb') as f:
                datafile = np.load(f, allow_pickle=True)
                data = {key: datafile[key] for key in datafile}
            
        stride = int(data['mocap_frame_rate'] // self.tgt_fps)
        if stride < 1:
            return None
        transl = torch.tensor(data['trans']).float()[::stride]
        glorot_aa = torch.tensor(data['root_orient']).float()[::stride]
        poses = torch.tensor(data['pose_body']).float()[::stride]
        jts = torch.tensor(data['jts_body']).float()[::stride]
        betas = torch.tensor(data['betas']).float()
        if self.shape_noise_sigma > 1e-6:
            betas += self.shape_noise_sigma * torch.randn_like(betas)
        betas = betas.repeat(transl.shape[0],1)
        
        if self.seq_len >=0:
            if transl.shape[0] <= self.seq_len: # remove very short sequences
                return None
            seq_start = np.random.randint(0,transl.shape[0]-self.seq_len)
            transl = transl[seq_start:seq_start + self.seq_len]
            glorot_aa = glorot_aa[seq_start:seq_start + self.seq_len]
            poses = poses[seq_start:seq_start + self.seq_len]
            betas = betas[seq_start:seq_start + self.seq_len]
            jts = jts[seq_start:seq_start + self.seq_len]

        else:
            nt = transl.shape[0]
            transl = transl[:nt//2*2]
            glorot_aa = glorot_aa[:nt//2*2]
            poses = poses[:nt//2*2]
            betas = betas[:nt//2*2]
        xb = torch.cat([transl, glorot_aa, poses],dim=-1)

        if self.reverse_time_dimension:
            xb = xb.flip(0)
            betas = betas.flip(0)
            jts = jts.flip(0)

        return {"betas":betas, "xb": xb, "jts_body": jts}




class AMASS_SMPLX_NEUTRAL_TEST(Dataset):
    def __init__(self, cfg):
        super().__init__()
        datapath = cfg["path"]
        subsets = cfg["subsets"]
        self.read_all_files_to_ram = cfg["read_all_files_to_ram"]
        self.tgt_fps = cfg["framerate"]
        self.seq_len = seq_len = cfg["seq_len"]

        self.npz_files = []
        for subset in subsets:
            self.npz_files += sorted(glob.glob(join(datapath,subset, "*/*.npz"),recursive=True))

        # different from the trainval set, here we trim the sequences first, so as to fix the testing motion segments
        self.all_seq = []
        for file in self.npz_files:
            data = np.load(file, allow_pickle=True)
            stride = int(data['mocap_frame_rate'] // self.tgt_fps)
            if stride < 1:
                continue
            
            trans = data['trans'][::stride]
            root_orient = data['root_orient'][::stride]
            pose_body = data['pose_body'][::stride]
            nt = trans.shape[0]
            t = 0

            while t < nt:
                # trans = data['trans'][::stride]
                # root_orient = data['root_orient'][::stride]
                # pose_body = data['pose_body'][::stride]
                if trans.shape[0] <= 2:
                    break
                if t+seq_len >= nt:
                    break
                data_ = {}
                data_['trans'] = trans[t:t+seq_len]
                data_['root_orient'] = root_orient[t:t+seq_len]
                data_['pose_body'] = pose_body[t:t+seq_len]
                data_['betas'] = data['betas']
                self.all_seq.append(data_)
                t += seq_len
        
    def __len__(self):
        return len(self.all_seq)
        
    def __getitem__(self, index):
        data = self.all_seq[index] 
        
        transl = torch.tensor(data['trans']).float()
        glorot_aa = torch.tensor(data['root_orient']).float()
        poses = torch.tensor(data['pose_body']).float()
        betas = torch.tensor(data['betas']).float().repeat(transl.shape[0],1)        
        
        xb = torch.cat([transl, glorot_aa, poses],dim=-1)

        return {"betas":betas, "xb": xb}
    







from primal.utils.data_io import *
class CustomizedActionMC(Dataset):
    """
    we collect some customized .smpl data from the meschapade platform.
    """
    def __init__(self, cfg):
        super().__init__()
        datapath = cfg["path"]
        self.seq_len = seq_len = cfg["seq_len"]
        self.tgt_fps = cfg["framerate"]
        
        # load files
        self.files = sorted(glob.glob(join(datapath, "*.smpl"),recursive=True))
        assert len(self.files) > 0, 'No files found in {}'.format(datapath)

        self.all_seq = []
        for file in self.files:
            # Assert file naming convention: [action]_[idx].smpl
            filename_base = os.path.basename(file).split('.')[0]
            assert '_' in filename_base, f"File {file} must follow naming convention [action]_[idx].smpl, but got {os.path.basename(file)}"
            assert len(filename_base.split('_')) >= 2, f"File {file} must follow naming convention [action]_[idx].smpl, but got {os.path.basename(file)}"
            
            betas_, xb_ = load_data_smplcodec(file, n_frames=None, tgt_fps=self.tgt_fps)
            ## trim to primitives here directly, in order to get action count also based on frames.
            action_label = filename_base.split('_')[0]
            nt = xb_.shape[0]
            tt = 0
            while tt < nt:
                if tt+self.seq_len > nt:
                    break
                data = {}
                data['action_label'] = action_label
                data['betas'] = betas_.copy()
                data['xb'] = xb_[tt:tt+self.seq_len].copy()
                self.all_seq.append(data)   
                tt += self.seq_len
            
        # gather action labels and their counts
        actions = [data['action_label'] for data in self.all_seq]
        self.action_counter = Counter(actions)
        self.action_label_list = list(self.action_counter.keys())
        print(self.action_counter)
        print(self.action_label_list)
        
    def get_samples_weights(self):
        sample_weights = []
        for sample in self.all_seq:
            # sum counts if multiple categories
            weight = 1.0/self.action_counter[sample['action_label']]
            sample_weights.append(weight)
        
        return torch.tensor(sample_weights).float()

    def __len__(self):
        return len(self.all_seq)

    def __getitem__(self, index):
        data = self.all_seq[index]
        betas, xb, action_label = data['betas'], data['xb'], data['action_label']
        betas = torch.tensor(data['betas']).float().repeat(xb.shape[0],1)
        xb = torch.tensor(xb).float()
        action_id = torch.tensor(self.action_label_list.index(action_label)).long().unsqueeze(0)

        return {"betas":betas, "xb": xb, "action_label": action_id}





import os
import sys
import glob

import numpy as np


import torch
from tqdm import tqdm

from primal.utils.eval_metrics import (
                                     compute_NRDF_dist, 
                                     compute_skating, 
                                     compute_noncollision,
                                     compute_mean_vel_err,
                                     compute_facing_dir_err, compute_target_err,
                                     compute_rprecision
                                     )


################ temporary code block to handle dip_goal results ###############
def dip_goal_find_tgt_locs(filename):
    if 'back_1.0' in filename:
        tgt_loc = np.array([0,0,-1])[None,None,:]
    elif 'front_1.0' in filename:
        tgt_loc = np.array([0,0,1])[None,None,:]
    elif 'left_1.0' in filename:
        tgt_loc = np.array([-1,0,0])[None,None,:]
    elif 'right_1.0' in filename:
        tgt_loc = np.array([1,0,0])[None,None,:]
    elif 'back_1.5' in filename:
        tgt_loc = np.array([0,0,-1.5])[None,None,:]
    elif 'front_1.5' in filename:
        tgt_loc = np.array([0,0,1.5])[None,None,:]
    elif 'left_1.5' in filename:
        tgt_loc = np.array([-1.5,0,0])[None,None,:]
    elif 'right_1.5' in filename:
        tgt_loc = np.array([1.5,0,0])[None,None,:]
    
    return tgt_loc





def evaluation(seqfile) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_
    """

    # 240 for motion realism evaluation
    # 90 for principled action generation and score-based guidance
    if 'MotionRealism' in seqfile:
        N_FRAMES = 240
    elif 'Principled' in seqfile:
        N_FRAMES = 90
    elif 'ARDiffusionSpatial' in seqfile:
        N_FRAMES = 60
    elif 'ARDiffusionAction' in seqfile:
        N_FRAMES = 90
    elif 'closd' in seqfile:
        N_FRAMES = 240
    else:
        raise Exception



    # placeholders and hparams
    skating_ratio = []
    noncoll_ratio = []
    dist_to_posemanifold = []    
    err_velmean = []
    err_facingdir = []
    err_tgtloc = []
    rprecisions = []
    
    with open(seqfile, 'rb') as f:
        data_all = pickle.load(f)
    
    if 'HumanEva' in seqfile:
        NB = 80
    elif 'SFU' in seqfile:
        NB = 160
    elif 'dip_t2m' in seqfile:
        NB=data_all['kpts'].shape[0]
        data_all2 = []
        for nb in range(NB):
            dd = {}
            dd['betas'] = data_all['betas'][nb:nb+1]
            dd['xb'] = data_all['xb'][nb:nb+1]
            dd['kpts'] = data_all['kpts'][nb:nb+1]
            data_all2.append(dd)
        data_all = data_all2.copy()
    elif 'dip_goal' in seqfile:
        NB=data_all['kpts'].shape[0]
        data_all2 = []
        for nb in range(NB):
            dd = {}
            dd['betas'] = data_all['betas'][nb:nb+1]
            dd['xb'] = data_all['xb'][nb:nb+1]
            dd['kpts'] = data_all['kpts'][nb:nb+1]
            dd['tgt_locs'] = dip_goal_find_tgt_locs(seqfile)
            data_all2.append(dd)
        data_all = data_all2.copy()
    elif 'closd' in seqfile:
        NB=data_all['kpts'].shape[0]
        data_all2 = []
        for nb in range(NB):
            dd = {}
            dd['betas'] = data_all['betas'][nb:nb+1]
            dd['xb'] = data_all['xb'][nb:nb+1]
            dd['kpts'] = data_all['kpts'][nb:nb+1]
            data_all2.append(dd)
        data_all = data_all2.copy()
        
    else:
        raise ValueError(f'Unknown dataset: {seqfile}')
    
    # print('kpts shape:', data_all[0]['kpts'].shape)
    for idx, data in tqdm(enumerate(data_all)):
        if idx >= NB:
            break

        if 'kpts' not in data.keys():
            raise ValueError('kpts is missing in the results.')
        else:
            xb = torch.from_numpy(data['xb'][0])[:N_FRAMES]
            kpts = torch.from_numpy(data['kpts'][0])[:N_FRAMES][...,:3]

        if 'tgt_vel' in data.keys():
            tgt_vel = torch.from_numpy(data['tgt_vel'])
        
        if 'tgt_locs' in data.keys():
            tgt_locs = torch.from_numpy(data['tgt_locs'])
        
        if 'action_label' in data.keys():
            tgt_action = data['action_label']
            xs = torch.from_numpy(data['xs']) # [b,T,d] a batch of canonicalized motion segments.


        ###########################################################################
        # 1. long-term motion generation
        # 1.1: NRDF pose naturalness
        # 1.2: foot-ground skating
        ###########################################################################
        ## 1.1: NRDF pose naturalness
        pose_aa = xb[:,6:69].reshape(-1, 63)
        dist = compute_NRDF_dist(pose_aa)
        dist_to_posemanifold.append(dist)

        ## 1.2: foot-ground skating
        ## get vertices at the feetbottom
        verts_feet = kpts[:,[10,11],:] # in case of jts
        skating_ratio_, contact_ratio_ = compute_skating(verts_feet, fps=30)
        skating_ratio.append(skating_ratio_)

        ## 1.3: noncollision ratio
        non_coll_ = compute_noncollision(kpts)
        noncoll_ratio.append(non_coll_)

        ###########################################################################
        # 2. principled control performances
        # 2.1: error of mean velocity
        # 2.2: error of facing direction
        ###########################################################################
        ## 2.1: error of mean velocity
        if 'tgt_vel' in data.keys():
            kpts_vel = (kpts[1:]-kpts[:-1])*30
            err_vel = compute_mean_vel_err(kpts_vel, tgt_vel,start_frames=30)
            err_velmean.append(err_vel)

            err_facingdir_ = compute_facing_dir_err(kpts, tgt_vel,start_frames=30)
            err_facingdir.append(err_facingdir_)
        
        if 'tgt_locs' in data.keys():
            ## the minimal distances
            err_tgtloc_ = compute_target_err(kpts - kpts[:1], 
                                             tgt_locs,
                                             goal_type='2D'
                                             )
            err_tgtloc.append(err_tgtloc_)

        if 'action_label' in data.keys():
            r_precision_ = compute_rprecision(xs, tgt_action)
            rprecisions.append(r_precision_)




    print(len(skating_ratio))
    # gather all quantiative results
    skating_ratio = np.mean(skating_ratio)
    dist_to_posemanifold_mean = np.mean(dist_to_posemanifold)
    noncoll_ratio = np.mean(noncoll_ratio)
    
    print()
    # print(f'results of : {os.path.basename(seqfile)}')
    print(f'results of : {seqfile}')
    print(f'--ASR (averaged skating ratio) = {skating_ratio:.5f}')
    print(f'--ANC (averaged non-collision ratio) = {noncoll_ratio:.5f}')
    print(f'--AND (averaged NRDF distance) = {1000*dist_to_posemanifold_mean:.5f} x 10^-3')
    if 'tgt_vel' in data.keys():
        print(f'--VelErr (averaged velocity error) = {np.mean(err_velmean):.5f}')
        print(f'--DirErr (averaged facing direction error) = {np.mean(err_facingdir):.5f}')
    if 'tgt_locs' in data.keys():
        print(f'--LocErr (averaged target location error) = {np.mean(err_tgtloc):.5f}')
    if 'action_label' in data.keys():
        print(f'--RP (R-precision, Top-1) = {np.mean(rprecisions):.5f}')

    print()




def evaluation_onefile(seqfiledir) -> None:
    # placeholders and hparams
    skating_ratio = []
    noncoll_ratio = []

    seqfiles = glob.glob(os.path.join(seqfiledir, '*.pkl'))
    for seqfile in tqdm(seqfiles):
        with open(seqfile, 'rb') as f:
            data = pickle.load(f)
            kpts = torch.from_numpy(data['kpts'][0])[...,:3]
            ###########################################################################
            # 2. long-term motion generation
            # 2.1: NRDF pose naturalness
            # 2.2: foot-ground skating
            ###########################################################################
            ## 2.2: foot-ground skating
            ## get vertices at the feetbottom
            verts_feet = kpts[:,[10,11],:] # in case of jts
            skating_ratio_, contact_ratio_ = compute_skating(verts_feet, fps=30)
            skating_ratio.append(skating_ratio_)
            
            ## 2.3: noncollision ratio
            non_coll_ = compute_noncollision(kpts)
            noncoll_ratio.append(non_coll_)

    print(len(skating_ratio))
    # gather all quantiative results
    skating_ratio = np.mean(skating_ratio)
    noncoll_ratio = np.mean(noncoll_ratio)
    
    print()
    print(f'results of : {os.path.basename(seqfile)}')
    print(f'--ASR (averaged skating ratio) = {skating_ratio:.5f}')
    print(f'--ANC (averaged non-collision ratio) = {noncoll_ratio:.5f}')
    print()



    
if __name__=='__main__':
    import numpy as np
    import pickle
    import glob
    import sys
    

    datapath = sys.argv[1]

    evaluation(datapath)
    
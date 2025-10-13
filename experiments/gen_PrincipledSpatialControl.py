import os
import glob
from omegaconf import OmegaConf
from primal.data import mixed_motion_dm

from primal.models.motion_diffuser import ARDiffusion
import numpy as np
import pickle

import torch
from tqdm import tqdm
import random
import argparse


def seed_all(seed=42):
    # Set seed for torch
    torch.manual_seed(seed)
    
    # If using CUDA, set the seed for it as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for Python's random module
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the function to lock the random seed
seed_all(0)





def predict_mop(args) -> None:
    
    results_all = []
    t_his = 1
    n_frames = args.n_frames

    # main loop
    for batch in tqdm(dataloader):
        for k,v in batch.items():
            if type(v) is not str:
                if v.ndim != 3:
                    v = v.unsqueeze(0)
                batch[k] = v.to(model.device)[:,:t_his+1]
            else:
                print(v)
        
        batch['betas'] = torch.randn_like(batch['betas'][:,:1]).repeat(1,t_his+1,1)
        
        batch["ori"] = ori

        betas, xb_gen, _, _ = model.generate_perpetual_navigation(
            batch, 
            n_inference_steps=model.hparams.scheduler.num_train_timesteps,
            nt_max = n_frames,
            snap_to_ground=True,
            reproj_kpts = args.use_reproj_kpts,
            switch_on_inertialization=args.use_inertialization,
            switch_on_control=True,
            guidance_weight_mv=50,
            guidance_weight_facing=25,
        )
        xb_gen = xb_gen[0]
        xb_gen_6d = model.aa2rotcont(xb_gen)
        betas = betas.repeat(1,xb_gen.shape[1],1)
        kpts_gen = model._fwd_smplx_seq(betas, xb_gen_6d,return_ssm2=False)

        result_one = {
            'betas':betas.detach().cpu().numpy(),
            'xb':xb_gen.detach().cpu().numpy(),
            'kpts':kpts_gen.detach().cpu().numpy(),
            'tgt_vel': ori.detach().cpu().numpy(),
        }
        results_all.append(result_one)
    
    outputdir = os.path.join('outputs', f'PrincipledSpatialControl/{os.path.basename(args.expdir)}')
    os.makedirs(outputdir, exist_ok=True)
    outputfile = os.path.join(
        outputdir,
        f'{args.dataset}_ema-{args.use_ema}_reproj-{args.use_reproj_kpts}_inertial-{args.use_inertialization}_dir-{args.target_dir}_speed-{args.target_speed}.pkl')
    
    with open(outputfile, 'wb') as handle:
        pickle.dump(results_all, handle)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    # about the checkpoint to eval
    parser.add_argument('--expdir', help='the training experiment directory', required=True)
    parser.add_argument('--ckptidx', help='the index (num of epoch) of the checkpoint', required=True)
    parser.add_argument("--use_ema", help="use the ema model parameters",action="store_true")
    # about the testing dataset
    parser.add_argument('--dataset', help='name of the evaluation dataset', required=True)
    parser.add_argument('--frame_skip', type=int, default=400,
                        help='the number of frames are skipped to obtain the initial body condition.')
    # about pre and post processing
    parser.add_argument("--use_reproj_kpts", 
                        help="preprocessing: reproject the keypoint locations to SMPL-X",
                        action="store_true")
    parser.add_argument("--use_inertialization",
                        help="postprocessing: inertialize to blend the motion seed and the generated motion",
                        action="store_true")
    # about action generation
    parser.add_argument("--target_dir", type=float,
                        help="the direction of the action,[0,90,180,270] degree in the world coordinate",
                        required=True)
    parser.add_argument("--target_speed", type=float,
                        help="the direction of the action",
                        required=True)
    parser.add_argument("--n_frames", type=int, default=100,)
    
    
    
    args = parser.parse_args()

    
    ###########################################################################
    # check IO
    ###########################################################################
    assert args.dataset in ['HumanEva', 'SFU'], 'Dataset must be HumanEva or SFU'


    ###########################################################################
    # setup motion diffusion model
    ###########################################################################
    print('loading testing data')
    ckptpaths = sorted(
        glob.glob(os.path.join(args.expdir, f'tensorboard/version_0/checkpoints/*={args.ckptidx}-*.ckpt')),
        key=os.path.getmtime
    )
    ## load the latest checkpoint
    ckptpath = ckptpaths[-1]
    print(f'loading model from {ckptpath}')
    model = ARDiffusion.load_from_checkpoint(
        ckptpath, 
        map_location='cuda:0',
        strict=False
    )
    
    if args.use_ema:
        model.load_ema_parameters()
    else:
        print('--not using ema model')
    model.eval()
    model.freeze()


    ###########################################################################
    # setup evaluation data
    ###########################################################################
    print('loading testing data')
    cfgdata = OmegaConf.load('primal/configs/data/amass_smplx.yaml')
    cfgdata.subsets=[args.dataset]
    cfgdata.batch_size=1
    cfgdata.seq_len = args.frame_skip
    dm = mixed_motion_dm.TestMotionDataModule(cfgdata)
    dataloader = dm.dataloader()
    
    ## define the traj. 
    ori_angle_deg = np.deg2rad(args.target_dir)
    ori = [np.cos(ori_angle_deg), 0, np.sin(ori_angle_deg)]
    ori = torch.tensor(ori)[None,None,...] #[b=1,t=1,3]
    ori = ori.float().to(model.device) * args.target_speed

    ## other params
    nt_tw = model.hparams.data.seq_len-1


    ###########################################################################
    # start evaluation
    ###########################################################################
    predict_mop(args)
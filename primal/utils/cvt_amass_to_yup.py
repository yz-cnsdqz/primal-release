"""AMASS is by default Z-up. 
This script is to transform the body parameters to Y-up.
"""

from mop_repr import SMPLXParser
import os.path as osp
import os, glob
import numpy as np
import torch
from tqdm import tqdm

# load raw amass sequence files
amass_path = 'datasets/AMASS/AMASS_SMPLX_NEUTRAL'
seqfiles = sorted(glob.glob(osp.join(amass_path, '*/*/*.npz')))

# load smplx parser
smplx_path = 'model-registry'
smplx_parser = SMPLXParser(osp.join(smplx_path,"models/SMPLX/neutral/SMPLX_neutral.npz"),
                                   osp.join(osp.dirname(osp.abspath(__file__)),"SSM2.json"),
                                   num_betas=16)
smplx_parser.eval()
smplx_parser.cuda()


# main loop
for seqfile in seqfiles:
    
    ## load data
    print(f'process: {seqfile}')
    try:
        data = np.load(seqfile)
        trans = torch.tensor(data['trans']).float().cuda() #[t,3]
    except:
        continue
    glorot_aa = torch.tensor(data['root_orient']).float().cuda()#[t,3]
    pose_body = torch.tensor(data['pose_body']).float().cuda() #[t,63]
    nt = trans.shape[0]
    betas = torch.tensor(data['betas']).float().cuda().repeat(nt,1) #[t,3]
    
    ## process data (transform to Y-up)
    ## to change Z-up to Y-up, we just need to rotate x axis by 90 deg
    transf_transl = torch.zeros(1,1,3).cuda() #[0,0,0]
    transf_rotmat = torch.tensor(
        [ [1.0000000,  0.0000000,  0.0000000],
          [0.0000000,  0.0000000, -1.0000000],
          [0.0000000,  1.0000000,  0.0000000 ]]
    ).unsqueeze(0).float().cuda()
    
    ## update smplx params
    xb = torch.cat([trans, glorot_aa,pose_body],dim=-1)
    xb_new = smplx_parser.update_transl_glorot(transf_rotmat, transf_transl, betas,xb)
    jts_body = smplx_parser.forward_smplx(betas, xb_new[:,:3], xb_new[:,3:6], xb_new[:,6:69],returntype='jts')[:,:22]

    ## save data
    trans_new = xb_new[:,:3].detach().cpu().numpy()
    glorot_aa_new = xb_new[:,3:6].detach().cpu().numpy()
    jts_body_np = jts_body.detach().cpu().numpy()
    output_data = {}
    output_data['mocap_frame_rate'] = data['mocap_frame_rate']
    output_data['trans'] = trans_new
    output_data['root_orient'] = glorot_aa_new
    output_data['jts_body'] = jts_body_np
    output_data['betas'] = data['betas']
    output_data['pose_body'] = data['pose_body']
    output_data['pose_hand'] = data['pose_hand']
    output_data['pose_jaw'] = data['pose_jaw']
    output_data['pose_eye'] = data['pose_eye']
    
    ## solve output path
    ## create dirs
    outputseqfile = seqfile.replace('AMASS_SMPLX_NEUTRAL', 'AMASS_SMPLX_NEUTRAL_YUP_wjts')
    outputseqdir = osp.dirname(outputseqfile)

    os.makedirs(outputseqdir, exist_ok=True)

    np.savez(outputseqfile, **output_data)



        
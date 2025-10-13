import os
import numpy as np
import random
from smplcodec import SMPLCodec
import pickle 



###########################################################################
# utility functions for file IO
###########################################################################
def load_data_npz(seqpath, random_segment=True, tgt_fps=30, n_frames=2):
    data = np.load(seqpath)
    stride = int(data['mocap_frame_rate'] // tgt_fps)
    transl = data['trans'][::stride]
    glorot_aa = data['root_orient'][::stride]
    betas = data['betas']
    pose_body = data['pose_body'][::stride]

    nt = transl.shape[0]
    if random_segment:
        tidx = random.randint(0, nt-n_frames)
    else:
        tidx = 0

    transl = transl[tidx:tidx+n_frames]
    glorot_aa = glorot_aa[tidx:tidx+n_frames]
    betas = data['betas']
    pose_body = pose_body[tidx:tidx+n_frames]
    xb = np.concatenate([transl, glorot_aa, pose_body], axis=-1)
    return betas, xb


def load_data_smplcodec(seqpath, n_frames=None, tgt_fps=30, return_dict=False):
    data = np.load(seqpath)
    
    #load shape params
    betas = data['shapeParameters']
    betas = betas[:16]
    if betas.shape[0] < 16:
        nbb = betas.shape[0]
        betas = np.pad(betas, (0,16-nbb), 'constant', constant_values=0)
    
    # load pose params
    if data['frameCount'] > 1:
        if data['frameRate'] <30:
            realfps = 30
        else:
            realfps = data['frameRate']
        stride = int(realfps // tgt_fps)
        transl = data['bodyTranslation'][::stride]
        glorot_aa = data['bodyPose'][::stride,0]
        pose_body = data['bodyPose'][::stride,1:]
        pose_body = pose_body.reshape(-1,63)
        if n_frames is not None:
            transl = transl[15:15+n_frames]
            glorot_aa = glorot_aa[15:15+n_frames]
            pose_body = pose_body[15:15+n_frames]
    elif data['frameCount'] == 1:
        transl = np.repeat(data['bodyTranslation'],2,axis=0)
        glorot_aa = np.repeat(data['bodyPose'],2,axis=0)[:,0]
        pose_body = np.repeat(data['bodyPose'],2,axis=0)[:,1:]
        pose_body = pose_body.reshape(-1,63)
    else:
        raise ValueError("data['frameCount'] should be >=1")
    
    if return_dict:
        return {
            'betas': betas,
            'transl': transl,
            'glorot_aa': glorot_aa,
            'pose_body': pose_body
        }
    else:
        xb = np.concatenate([transl, glorot_aa, pose_body], axis=-1)
        return betas, xb



def load_data(seqpath, tgt_fps=30, n_frames=2):
    ext = os.path.splitext(seqpath)[1]
    if ext == '.npz':
        betas, xb = load_data_npz(
            seqpath, 
            random_segment=True,
            n_frames=n_frames,
            )
    elif ext == '.smpl':
        betas, xb = load_data_smplcodec(seqpath, n_frames=2, 
                                        tgt_fps=tgt_fps)
    else:
        raise NotImplementedError
        
    return betas, xb



def save_data_pkl(outputdir, idx, betas, xb, kpts):
    """
    this pkl file is compatible the headless rendering
    """
    os.makedirs(outputdir, exist_ok=True)
    outfilepath = os.path.join(outputdir,f'result_{idx:08d}.pkl')
    betas_np = betas.detach().cpu().numpy()
    xb_np = xb.detach().cpu().numpy()
    kpts_np = kpts.detach().cpu().numpy()
    output={}
    output['betas'] = betas_np
    output['xb'] = xb_np
    output['kpts'] = kpts_np
    with open(outfilepath, 'wb') as handle:
        pickle.dump(output, handle)
    
    return outfilepath



def save_data_smplcodec(outputdir, idx, betas, xb, tgt_fps=30):
    """
    this pkl file is compatible the headless rendering
    """
    outfilepath = os.path.join(outputdir,f'result_{idx:08d}.smpl')
    betas_np = betas.detach().cpu().numpy()[0,0]
    xb_np = xb.detach().cpu().numpy()[0]
    nt = xb_np.shape[0]

    codec1 = SMPLCodec()
    codec1.shape_parameters = betas_np
    codec1.frame_count = nt
    codec1.frame_rate = float(tgt_fps)
    codec1.body_translation = xb_np[:,:3]  # [N x 3] Global trans
    glorot_aa = xb_np[:,3:6][:,None,:]
    pose_aa = xb_np[:,6:].reshape(-1,21,3)
    codec1.body_pose = np.concatenate([glorot_aa, pose_aa],axis=1) # [N x 22 x 3] pelvis..right_wrist
    codec1.head_pose = np.zeros((nt,3,3))  # [N x 3 x 3] jaw, leftEye, rightEye
    codec1.left_hand_pose = np.zeros((nt,15,3)) # [N x 15 x 3] left_index1..left_thumb3
    codec1.right_hand_pose = np.zeros((nt,15,3)) # [N x 15 x 3] right_index1..right_thumb3

    codec1.write(outfilepath)
    return outfilepath




def save_data_npz(outputdir, 
                  idx, betas, xb, 
                  tgt_fps=30):
    """
    this npz file is compatible with the AMASS format
    """
    outfilepath = os.path.join(outputdir,f'result_{idx:08d}.npz')
    betas_np = betas.detach().cpu().numpy()[0,0]
    xb_np = xb.detach().cpu().numpy()[0]
    nt = xb_np.shape[0]

    # Extract data components from xb
    trans = xb_np[:,:3]  # (nt, 3) - translation
    root_orient = xb_np[:,3:6]  # (nt, 3) - root orientation
    pose_body = xb_np[:,6:].reshape(-1,21,3)  # (nt, 21, 3) - body pose
    
    # Create joint positions (placeholder - would need proper forward kinematics)
    jts_body = np.zeros((nt, 22, 3))  # (nt, 22, 3)
    

    # Create hand, jaw, and eye poses (zeros as placeholders)
    pose_hand = np.zeros((nt, 90))  # (nt, 90) - hand poses (45 per hand)
    pose_jaw = np.zeros((nt, 3))   # (nt, 3) - jaw pose
    pose_eye = np.zeros((nt, 6))   # (nt, 6) - eye poses (3 per eye)
    
    # Save in NPZ format with AMASS structure
    np.savez(outfilepath,
             mocap_frame_rate=tgt_fps,
             trans=trans,
             root_orient=root_orient,
             jts_body=jts_body,
             betas=betas_np,
             pose_body=pose_body.reshape(nt, -1),  # Flatten to (nt, 63)
             pose_hand=pose_hand,
             pose_jaw=pose_jaw,
             pose_eye=pose_eye)
    
    return outfilepath
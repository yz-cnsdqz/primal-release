"""
modified from MOJO
https://github.com/yz-cnsdqz/MOJO-release/blob/master/experiments/utils/eval_metrics.py
"""


import sys
import os
import glob
import torch
import torch.nn.functional as F

############### load NRDF to measure pose realism ###############
## need to install NRDF first
NRDF_path = os.getenv("NRDF_PATH", "/home/yan/NRDF")
sys.path.insert(0, os.path.abspath(NRDF_path))
try:
    from nrdf.exp.pose_metric import NRDFProjector
    projector = NRDFProjector()
except:
    print('cannot load NRDF pose metric')
    projector = None



def compute_NRDF_dist(pose_aa:torch.Tensor):
    """
    pose_aa: [B, 63]
    """
    if projector is None:
        return -1.0
    projector = NRDFProjector(device=pose_aa.device)
    dist = projector.predict_dist(pose_aa).mean().item()
    return dist

##############################################################



###########################################################################
# this is specifically for semantic action generation evaluation
# one needs to adapt the code (e.g. number classes) for specific scenarios.
# setup the trained motion clip model.
# this model is only trained with CustomizedActionMC
###########################################################################
from primal.models.motionclip import MotionCLIP

try:
    ckptpaths = sorted(
        glob.glob(os.path.join('logs/motion_clip/runs/2025-02-24_15-40-20', 
                                'tensorboard/version_0/checkpoints/*=*-*.ckpt')),
        key=os.path.getmtime
    )
    ## load the latest checkpoint
    ckptpath = ckptpaths[-1]
    print(f'loading model from {ckptpath}')

    cmotionclip = MotionCLIP.load_from_checkpoint(
        ckptpath, 
        map_location='cuda:0',
        strict=False
    )

    cmotionclip.eval()
    cmotionclip.freeze()
except:
    print('cannot load the pretrained motion clip model')
    cmotionclip = None

def compute_rprecision(xs, action_label):
    """
    note that this top-1 R-precision is equivalent to recognition accuracy
    xs: [b,T,d], the canonicalized motion segments
    action_label: [b,1,1] their ground truth action labels 
    """
    if cmotionclip is None:
        return -1.0
    
    xs = xs.to(cmotionclip.device)
    nb = xs.shape[0]
    action_all = torch.arange(5).long().to(xs.device) #[b=1,5]
    # extract features
    xs_emb = cmotionclip.encode_motion(xs) #[b,d]
    # action_emb = cmotionclip.encode_text(action_label)#[b,d]
    action_all_emb = cmotionclip.encode_text(action_all)#[5,d]
    
    # compute the r-precision top-1
    D = torch.cdist(xs_emb.unsqueeze(0), action_all_emb.unsqueeze(0), p=2)[0] #[b,5]

    action_label_pred = torch.argmin(D,dim=-1) # [b,] we then should have labels
    action_label = action_label[:,0]
    acc = (action_label_pred==action_label.item()).sum()/nb

    return acc.item()

###########################################################################



def compute_target_err(kpts_pred,tgt_locs, goal_type='3D'):
    """
    compute the l2 distance between the target pelvis location 
        and the predicted pelvis location in the last frame

    kpts_pred: the predicted joint velocities, [T, J, 3]
    tgt_locs: the ground truth joint velocities, [1, 1, 3]
    start_frames: the number of frames to compute the mean velocity distance

    """
    assert kpts_pred.ndim == 3, "the shape of vels_pred should be (T, J, 3)"
    assert tgt_locs.shape == (1,1,3), "the shape of vels_mean_tgt should be (1,1,3)"
    if goal_type == '3D':
        pelvis_locs = kpts_pred[:,0] #[T,3]
        tgt_locs = tgt_locs[0] #[1,3]    
    elif goal_type == '2D':
        pelvis_locs = kpts_pred[:,0,[0,2]] #[T,2]
        tgt_locs = tgt_locs[:,0,[0,2]] #[1,2]
    else:
        raise NotImplementedError
    
    
    err = (pelvis_locs - tgt_locs).norm(dim=-1).amin()
    return err



def compute_mean_vel_err(vels_pred,vels_mean_tgt, start_frames=0):
    """
    vels_mean_tgt: the ground truth joint velocities, [1, 1, 3]
    vels_pred: the predicted joint velocities, [T, J, 3]
    start_frames: the number of frames to compute the mean velocity distance

    Returns: the l2 norm
    """
    assert vels_mean_tgt.shape == (1,1,3), "the shape of vels_mean_tgt should be (1,1,3)"
    assert vels_pred.ndim == 3, "the shape of vels_pred should be (T, J, 3)"

    vels_pred_mean = vels_pred[start_frames:].mean(dim=[0,1],keepdim=True)
    err = (vels_pred_mean - vels_mean_tgt).norm(dim=-1).mean()
    return err


def compute_facing_dir_err(kpts_pred, ori_tgt, start_frames=0):
    """
    kpts_pred: the predicted joint locations, [T, J, 3]
    ori_tgt: the ground truth facing direction, [1, 1, 3]
    start_frames: the number of frames to compute the mean velocity distance

    Returns: the l2 norm 
    """
    assert kpts_pred.ndim == 3, "the shape of kpts_pred should be (T, J, 3)"
    assert ori_tgt.shape == (1,1,3), "the shape of ori_tgt should be (1,1,3)"

    # compute facing directions at individual frames
    kpts_pred = kpts_pred[start_frames:]
    x_axis = kpts_pred[:,1] - kpts_pred[:,2]
    x_axis[...,1] = 0
    x_axis = F.normalize(x_axis, dim=-1).unsqueeze(-2)
    y_axis = torch.tensor([[[0,1,0]]]).float().to(kpts_pred.device)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
                    
    # process ori_tgt
    ori_tgt = F.normalize(ori_tgt, dim=-1).repeat(kpts_pred.shape[0],1,1)

    # compute facing direction error
    err = (z_axis-ori_tgt).norm(dim=-1).mean()

    return err





def compute_skating(
        kpts_feet: torch.Tensor,
        thresh_height: float = 5e-2,
        thresh_vel: float = 5e-3,
        fps: float = 30.0,
    ):
    """
    compute foot skating ratio provided the sequence of keypoint locations.
    Assuming the ground plane is XZ=0

    kpts_feet: [T, 2, 3]. The joint locations of the left and right foot.
    thresh_height: <= thresh_height means in contact.
    thresh_vel: vel_xz <= thresh_vel whereas in contact means no skating. This value is for time differences.
    fps: the sequence fps.
    """
    
    thresh_vel = thresh_vel * fps
    
    verts_feet_horizon_vel = (kpts_feet[1:,:,[0,2]]-kpts_feet[:-1,:,[0,2]])*fps
    verts_feet_horizon_vel = torch.norm(verts_feet_horizon_vel, dim=-1)
    verts_feet_height = kpts_feet[1:,:,1].abs()
    contact_labels = verts_feet_height<thresh_height
    nt = verts_feet_height.shape[0]

    skating = (verts_feet_horizon_vel>thresh_vel)*contact_labels
    skating_ratio = torch.sum(skating[:,0] & skating[:,1]) / nt
    contact_ratio = torch.sum(contact_labels[:,0] & contact_labels[:,1]) / nt

    return skating_ratio.item(), contact_ratio.item()




def compute_noncollision(
        kpts: torch.Tensor,
        thresh_height: float = 5e-2,
    ):
    """
    compute noncollission issue with the ground. 
    Collision is regarded to be True, if a joint is below some thresh_height
    
    """    
    nt = kpts.shape[0]
    kpts_height = kpts[...,1] # the height values
    kpts_height_min = kpts_height.amin(dim=-1) # for each frame, we compute the lowest joint height, [t, 1]
    
    noncoll_ratio = 1. - torch.sum(kpts_height_min < -thresh_height) / nt

    return noncoll_ratio.item()


import torch
import torch.nn as nn
from human_body_prior.body_model.body_model import BodyModel
import json
import primal.utils.pytorch3d_transforms as p3dt
from primal.utils.joint_matching import SMPLX_SURFACE_KEYPOINTS
from primal.utils.body_models import SMPLXROTMAT



def primalsmpl(mop_transfs, bm):
    transfs = torch.zeros(mop_transfs.shape[0],mop_transfs.shape[1],4,4).float().to(mop_transfs.device)
    transfs[:,:,:3,:3] = p3dt.rotation_6d_to_matrix(mop_transfs[:,:,:6])
    transfs[:,:,:3,3] = mop_transfs[:,:,6:]
    poses_angles = torch.zeros(transfs.shape[0],transfs.shape[1],3).float().to(mop_transfs.device)
    
    for j in range(21,0,-1):
        poses_angles[:,j] = p3dt.matrix_to_axis_angle(torch.matmul(torch.inverse(transfs[:,bm.kintree_table[0,j],:3,:3]),transfs[:,j,:3,:3]))
    poses_angles[:,0] = p3dt.matrix_to_axis_angle(transfs[:,0,:3,:3])

    joints = bm.forward(root_orient=poses_angles[:,0],pose_body=poses_angles.view(poses_angles.shape[0],22*3)[:,3:]).Jtr

    trans = transfs[:,0,:3,3] - joints[:,0]

    return torch.cat([trans,poses_angles.reshape(trans.shape[0],22*3)],dim=1)

def get_norm_poses(trans, poses):
    """_summary_

    Args:
        trans (_type_): SMPL translation params (N x 3)
        poses (_type_): SMPL pose parameters (N x J*3)

    Returns:
        _type_: Normalized translation and pose parameters such that origin is ground projection of SMPL origin
    """
    poses_matrix = p3dt.axis_angle_to_matrix(poses.view([poses.shape[0],-1,3]))
    
    fwd = poses_matrix[0,0,:3,2].clone() 
    fwd[2] = 0
    fwd /= torch.linalg.norm(fwd)
    if fwd[0] > 0:
        tfm = p3dt.axis_angle_to_matrix(torch.tensor([0,0,torch.arccos(fwd[1])]).type_as(fwd).unsqueeze(0))
    else:
        tfm = p3dt.axis_angle_to_matrix(torch.tensor([0,0,-torch.arccos(fwd[1])]).type_as(fwd).unsqueeze(0))

    tfmd_orient = torch.einsum("ij,bjk->bik",tfm[0],poses_matrix[:,0])
    tfmd_trans = torch.einsum("ij,bj->bi",tfm[0],trans) 
    
    poses_matrix[:,0] = tfmd_orient
    
    tfmd_trans[:,:2] = tfmd_trans[:,:2] - tfmd_trans[0,:2]
    
    norm_poses = torch.cat([tfmd_trans,p3dt.matrix_to_axis_angle(poses_matrix).reshape(tfmd_trans.shape[0],-1)],dim=1)
    return norm_poses




"""
inertialization to merge two motion sequences.
- the first sequence and the second sequence have one frame overlap.
- the first sequence has a single frame.
"""

def inertial_offset(d0, d_dot0, t, omega):
    """
    Compute the inertialization offset for each frame using vectorized operations.
    """
    return (d0 + (d_dot0 + omega * d0) * t) * torch.exp(-omega * t)

def inertial_offset_derivative(d0, d_dot0, t, omega):
    """
    Compute the time derivative of the inertial offset for each frame using vectorized operations.
    """
    return torch.exp(-omega * t) * (d_dot0 - omega * (d_dot0 + omega * d0) * t)

def inertialize(seq1, seq2, omega=10.0, dt=1.0/30.0):
    """
    Inertialize the transition from a single frame seq1 to a multi-frame seq2 using PyTorch for parallel computation.
    
    Parameters:
      - seq1: a tensor representing the starting state (1 frame), [b,t,d]
      - seq2: representing the target motion, shape [b,t,d]
      - omega: a positive scalar controlling the decay (blend speed)
      - dt: time step per frame (in seconds), 30fps by default
      
    Returns:
      - new_seq2: a tensor of shape [b,t,d] with the inertial transition applied
    """
    # Convert seq1 and seq2 to PyTorch tensors
    # seq1 = torch.tensor(seq1, dtype=torch.float32)
    # seq2 = torch.tensor(seq2, dtype=torch.float32)
    assert seq1.shape[1] == 1, 'the first sequence should be single frame. Other cases are not implemented yet.'
    nb, nt = seq2.shape[:2]

    # Compute initial difference at t=0 (d0)
    d0 = seq2[:,:1] - seq1
    
    # For velocity, assume zero if not provided.
    d_dot0 = torch.zeros_like(d0)
    
    # Compute time indices for all frames in seq2
    time_indices = dt * torch.arange(nt, dtype=torch.float32).to(seq2.device)
    time_indices = time_indices[None,:,None]

    # Vectorize inertialization computation for all frames
    offsets = inertial_offset(d0, d_dot0, time_indices, omega)
    # Compute the corrected frames by subtracting the offsets
    new_seq2 = seq2 - offsets  # Add an extra dimension to align with seq2's dimensions

    return new_seq2
    
def inertialize_seq_his(seq1, seq2, omega=10.0, dt=1.0/30.0):
    """
    Inertialize the transition from seq1 (a multi-frame sequence) to seq2 (another multi-frame sequence) 
    by modifying seq2 to ensure a smooth transition.

    Parameters:
      - seq1: a tensor representing the starting sequence, shape [b,t1,d]
      - seq2: a tensor representing the target motion, shape [b,t2,d]
      - omega: a positive scalar controlling the decay (blend speed)
      - dt: time step per frame (in seconds), default 30fps

    Returns:
      - modified_seq2: a tensor of shape [b, t2, d] with inertialized transition
    """
    assert seq1.shape[1] > 1, 'seq1 must have multiple frames.'
    assert seq2.shape[1] > 1, 'seq2 must have multiple frames.'

    b, t1, d = seq1.shape
    _, t2, _ = seq2.shape

    # Estimate velocity from the last two frames of seq1
    d_dot0 = seq1[:, -1:] - seq1[:, -2:-1]  # Approximate velocity

    # Extrapolate the expected position of seq1 at t+1
    extrapolated_seq1 = seq1[:, -1:] + d_dot0  # Shape [b, 1, d]

    # Compute the difference (discontinuity) between the extrapolated frame and seq2's first frame
    d0 = seq2[:, :1] - extrapolated_seq1  # Offset needed to correct seq2

    # Compute time indices for seq2 frames (starting from t+1)
    time_indices = dt * torch.arange(t2, dtype=torch.float32).to(seq2.device)
    time_indices = time_indices[None, :, None]  # Shape [1, t2, 1]

    # Compute inertialized offsets
    offsets = inertial_offset(d0, d_dot0, time_indices, omega)

    # Modify seq2 to smoothly align with seq1
    modified_seq2 = seq2 - offsets

    # Combine seq1 and modified seq2
    combined_seq = torch.cat([seq1, modified_seq2], dim=1)

    return combined_seq





""" motion representation format based on GAMMA 
- assumes SMPLX-neutral
- assumes ssm2-67 body surface markers
"""
class RotConverter(nn.Module):
    '''
    - this class is modified from smplx/vposer
    - all functions only support data_in with [N, num_joints, D].
        -- N can be n_batch, or n_batch*n_time
    '''
    def __init__(self):
        super(RotConverter, self).__init__()

    @staticmethod
    def cont2rotmat(data_in):
        '''
        :data_in [...,6]
        :return: [...,3,3]
        '''
        output = p3dt.rotation_6d_to_matrix(data_in)
        return output

    @staticmethod
    def aa2cont(data_in):
        '''
        :data_in [...,3]
        :return: [...,6]
        '''
        pose_body_rotmat = p3dt.axis_angle_to_matrix(data_in)
        pose_body_6d = p3dt.matrix_to_rotation_6d(pose_body_rotmat)
        return pose_body_6d

    @staticmethod
    def rotmat2cont(data_in):
        '''
        :data_in [...,3,3]
        :return: [...,6]
        '''
        pose_body_6d = p3dt.matrix_to_rotation_6d(data_in)
        return pose_body_6d




    @staticmethod
    def cont2aa(data_in):
        '''
        :data_in [...,6]
        :return: [...,3]
        '''
        x_matrot = RotConverter.cont2rotmat(data_in)
        x_aa = RotConverter.rotmat2aa(x_matrot)
        return x_aa


    @staticmethod
    def rotmat2aa(data_in):
        '''
        :data_in data_in: [...,3,3]
        :return: [...,3]
        '''
        pose = p3dt.matrix_to_axis_angle(data_in).contiguous()
        return pose

    @staticmethod
    def aa2rotmat(data_in):
        '''
        :data_in [...,3]
        :return: [...,3,3]
        '''
        output = p3dt.axis_angle_to_matrix(data_in)
        return output




class CanonicalCoordinateExtractor(nn.Module):
    """Summary of class here.

    When the model runs recursively, we need to reset the coordinate and perform canonicalization on the fly.
    This class provides such functionality.
    When specifying the joint locations of the motion primitive, it produces a new canonical coordinate, according to
    the reference frame.
    Both numpy and torch are supported.

    Attributes:
        device: torch.device, to specify the device when the input is torch.tensor
    """

    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, jts: torch.Tensor
        ):
        """get a new canonical coordinate located at a specific frame
        NOTE: Compared to AMASS, the canonical coordinate is **Y-UP**.

        Args:
            jts: the input joint locations, jts.shape=[b,J,3]
            in_numpy: if True, everything is calculated with numpy, else calculated with pytorch.

        Returns:
            A list of tensors, [new_rotmat, new_transl]
            new_rotmat: rotmat according to the old coordinate. in the shape [b,3,3]
            new_transl: translation according to the old coordinate. in the shape [b,1,3]

        Raises:
            None
        """
        x_axis = jts[:,1,:] - jts[:,2,:] #[b,3],pointing from right to left
        x_axis[:, 1] = 0 # parallel to the XZ plane
        x_axis = x_axis / torch.norm(x_axis,dim=-1, keepdim=True)
        y_axis = torch.tensor([[0,1,0]]).float().to(jts.device).repeat(x_axis.shape[0], 1)
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        z_axis = z_axis/torch.norm(z_axis,dim=-1, keepdim=True)
        new_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=-1) #[b,3,3]
        new_transl = jts[:, :1] #[b,1,3] at the pelvis
        return new_rotmat, new_transl




class SMPLXParser(nn.Module):
    """operations about the smplx model

    We frequently use the smplx model to extract the markers/joints, and perform relevant transform.
    So we make this smplxparser to provide an unified interface.

    Attributes:
        an example input:
        pconfig_mp = {
            'n_batch':10,
            'device': device,
            'marker_placement': 'ssm2_67'
            }
    """
    def __init__(self, body_model_path,
                 marker_idx_path,
                 num_betas=10):
        super().__init__()

        '''set body models'''
        self.bm = BodyModel(body_model_path, model_type="smplx",num_betas=num_betas)
        with open(marker_idx_path) as f:
            self.marker_idx = list(json.load(f)['markersets'][0]['indices'].values())        

        self.bm.eval()
        self.coord_extractor = CanonicalCoordinateExtractor()


    def forward_smplx(self,
                    betas: torch.Tensor,
                    transl: torch.Tensor,
                    glorot_aa: torch.Tensor,
                    poses: torch.Tensor,
                    returntype: str = 'jts_and_ssm2_markers',
        ):
        """forward kinematics for smplx.

        Args:
            betas: the body shape, e.g. [b,10]
            transl: [b,3]
            glorot_aa: [b,3]
            poses: [b, 63]
            only_return_jts: bool

        Returns:
            output: np.array or torch.tensor, either joints [b, 22, 3] or markers [b, n_markers, 3]

        Raises:
            NotImplementedError rises if output_type is neither 'markers' nor 'joints'
        """

        smplxout = self.bm(trans=transl, root_orient=glorot_aa, betas=betas, pose_body=poses)
        jts = smplxout.Jtr
        if returntype == 'jts':
            return jts
        elif returntype == 'verts':
            return smplxout.v
        elif returntype == 'jts_body':
            return jts[:,:22]
        elif returntype == 'jts_and_ssm2_markers':
            markers = smplxout.v[:,self.marker_idx]
            return jts, markers
        elif returntype == 'jts_and_ssm2_markers_and_eyeverts':
            markers = smplxout.v[:,self.marker_idx]
            eyeverts = smplxout.v[:,[9929, 9448]]
            return jts, markers,eyeverts 
        elif returntype == 'jts_and_landmarks':
            ## 45 keypoints in total
            ## here we actually mix the joints and some landmarkers on the body surface to jts
            ## num_joints=55, following https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
            ## some additional landmarks: https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py
            ## here we only add [nose, reye, leye, rear, lear, ]
            # jts_body = jts[:,:22]
            # jts_hand_joints = jts[:,25:] # 
            landmarkids = list(SMPLX_SURFACE_KEYPOINTS.values())
            landmarks_extra = smplxout.v[:,landmarkids] 
            keypoints = torch.cat([jts, landmarks_extra], dim=-2)
            return keypoints
        else:
            raise NotImplementedError



    def get_new_coordinate(self,
                betas: torch.Tensor,
                xb: torch.Tensor,
                coordinate_on_ground=False,
        ):
        """get a batch canonical coordinates for recursive use.
        Note that this function extends self.coord_extractor.get_new_coordinate, and assumes the first frame is the reference frame.
        Therefore, it is required to select the motion seed before using this function.

        Args:
            betas: the body shape, [b,10]
            xb: the compact vector of body params, (b, d)
            
        Returns:
            [new_rotmat, new_transl]: np.array or torch.tensor, in the shape of [b, 3, 3] and [b, 1, 3]

        Raises:
            None
        """
        assert betas.ndim==2, 'incorrect betas shape, should be [b,d]'
        transl = xb[...,:3]
        glorot_aa = xb[...,3:6]
        poses_aa = xb[...,6:69]
        joints = self.forward_smplx(betas, transl, glorot_aa, poses_aa, returntype='jts')
        new_rotmat, new_transl = self.coord_extractor(joints)

        if coordinate_on_ground:
            new_transl[...,1] = 0 # only feasible when Y-up


        return new_rotmat, new_transl


    @torch.no_grad()
    def get_contact_features(self, contactidx, on_markers=True):
        """based on contact idx, 

        Args:
            contactidx: [b,] the mesh verts idx on smplx
            on_markers: if true, we only consider the markerset
        """
        assert on_markers, 'only support on_markers now...'

        marker_idx_tensor = torch.tensor(self.marker_idx)
        contact_labels_on_marker = []
        for cc in contactidx:
            cc = torch.tensor(cc)
            ccm = torch.isin(marker_idx_tensor, cc)
            contact_labels_on_marker.append(ccm)
        
        contact_labels_on_marker = torch.stack(contact_labels_on_marker)
        return contact_labels_on_marker.float()



    def calc_calibrate_offset(self,
                            betas: torch.Tensor,
                            body_pose: torch.Tensor,
        ):
        """compensate the offset when transforming the smplx body and the body parameterss
        When performing the global transformation, first getting the body and then transforming will lead to a different result
        from first transforming the transl/global_orient and then get the body model. The reason is the global rotation is about
        the pelvis, whereas the global translation is about the kinematic tree root, which is not the body pelvis.

        Args:
            betas: [b,10]
            body_pose: the body pose in axis-angle, in shape (b, 63)
            
        Returns:
            delta_T: the compensation offset between root and body pelvis, in shape (b, 3)

        Raises:
            None
        """
        smplx_out = self.bm(pose_body=body_pose, betas=betas)
        delta_T = smplx_out.Jtr[:,0,:] # we output all pelvis locations
        
        return delta_T

    @torch.no_grad()
    def update_transl_glorot(self,
                            transf_rotmat: torch.Tensor,
                            transf_transl: torch.Tensor,
                            betas: torch.Tensor,
                            xb: torch.Tensor,
                            fwd_transf: bool=False):
        """update the (global) body parameters when performing global transform
        When performing the global transformation, first getting the body and then transforming will lead to a different result
        from first transforming the transl/global_orient and then get the body model. The reason is the global rotation is about
        the pelvis, whereas the global translation is about the kinematic tree root, which is not the body pelvis.

        Args:
            transf_rotmat: torch.tensor or np.array in shape (b, 3, 3)
            transf_transl: torch.tensor or np.array in shape (b, 1, 3)
            betas: np.array or torch.tensor, in shape (10, ) or (1, 10)
            xb: torch.tensor or np.array in shape (b, 3+3+63), the smplx body parameters
            fwd_transf: if yes, apply (transf_rotmat, transf_transl) to xb; if false, transform xb to the coordinate (transf_rotmat, transf_transl)

        Returns:
            xb: the body parameters containing the updated transl and global_orient, (b, 3+3+63)

        Raises:
            None
        """
        
        delta_T = self.calc_calibrate_offset(betas,xb[:,6:69])
        transl = xb[:,:3]
        glorot = xb[:,3:6]
        
        if not fwd_transf:
            global_ori = RotConverter.aa2rotmat(glorot)
            global_ori_new = torch.einsum('bij,bjk->bik', transf_rotmat.permute(0,2,1), global_ori)
            glorot = RotConverter.rotmat2aa(global_ori_new)
            transl = torch.einsum('bij,bj->bi', transf_rotmat.permute(0,2,1), transl+delta_T-transf_transl[:,0])-delta_T
        else:
            global_ori = RotConverter.aa2rotmat(glorot)
            global_ori_new = torch.einsum('bij,bjk->bik', transf_rotmat, global_ori)
            glorot = RotConverter.rotmat2aa(global_ori_new)
            transl = torch.einsum('bij,bj->bi', transf_rotmat, transl+delta_T)-delta_T+transf_transl[:,0]

        xb[:,:3] = transl
        xb[:,3:6] = glorot

        return xb
    


    @torch.no_grad()
    def update_transl_glorot_seq(self,
                            transf_rotmat: torch.Tensor,
                            transf_transl: torch.Tensor,
                            betas: torch.Tensor,
                            xb_in: torch.Tensor,
                            fwd_transf: bool=False):
        """update the (global) body parameters when performing global transform
        When performing the global transformation, first getting the body and then transforming will lead to a different result
        from first transforming the transl/global_orient and then get the body model. The reason is the global rotation is about
        the pelvis, whereas the global translation is about the kinematic tree root, which is not the body pelvis.

        Args:
            transf_rotmat: torch.tensor or np.array in shape (b, 3, 3)
            transf_transl: torch.tensor or np.array in shape (b, 1, 3)
            betas: np.array or torch.tensor, in shape [b,t,d]
            xb: torch.tensor or np.array in shape (b, t, 3+3+63), the smplx body parameters
            fwd_transf: if yes, apply (transf_rotmat, transf_transl) to xb; if false, transform xb to the coordinate (transf_rotmat, transf_transl)

        Returns:
            xb: the body parameters containing the updated transl and global_orient, (b, t, 3+3+63)

        Raises:
            None
        """
        xb = xb_in.detach().clone()
        nb,nt = xb.shape[:2]
        delta_T_reshaped = self.calc_calibrate_offset(betas.reshape(nb*nt,-1),xb[:,:,6:69].reshape(nb*nt,-1))
        delta_T = delta_T_reshaped.reshape(nb,nt,-1)
        transl = xb[:,:,:3]
        glorot = xb[:,:,3:6]
        if not fwd_transf:
            global_ori = RotConverter.aa2rotmat(glorot)
            global_ori_new = torch.einsum('bij,btjk->btik', transf_rotmat.permute(0,2,1), global_ori)
            glorot_new = RotConverter.rotmat2aa(global_ori_new)
            transl_new = torch.einsum('bij,btj->bti', transf_rotmat.permute(0,2,1), transl+delta_T-transf_transl)-delta_T
        else:
            global_ori = RotConverter.aa2rotmat(glorot)
            global_ori_new = torch.einsum('bij,btjk->btik', transf_rotmat, global_ori)
            glorot_new = RotConverter.rotmat2aa(global_ori_new)
            transl_new = torch.einsum('bij,btj->bti', transf_rotmat, transl+delta_T)-delta_T+transf_transl

        xb[:,:,:3] = transl_new
        xb[:,:,3:6] = glorot_new

        return xb




class SMPLXParserRotcont(nn.Module):
    """operations about the smplx model

    We frequently use the smplx model to extract the markers/joints, and perform relevant transform.
    So we make this smplxparser to provide an unified interface.

    Attributes:
        an example input:
        pconfig_mp = {
            'n_batch':10,
            'device': device,
            'marker_placement': 'ssm2_67'
            }
    """
    def __init__(self, body_model_path,
                 marker_idx_path,
                 num_betas=10):
        super().__init__()

        '''set body models'''
        self.bm = SMPLXROTMAT(body_model_path, model_type="smplx",num_betas=num_betas)
        with open(marker_idx_path) as f:
            self.marker_idx = list(json.load(f)['markersets'][0]['indices'].values())        

        self.bm.eval()
        self.coord_extractor = CanonicalCoordinateExtractor()


    def forward_smplx(self,
                    betas: torch.Tensor,
                    transl: torch.Tensor,
                    glorot: torch.Tensor,
                    poses: torch.Tensor,
                    returntype: str = 'jts_and_ssm2_markers',
        ):
        """forward kinematics for smplx.

        Args:
            betas: the body shape, e.g. [b,10]
            transl: [b,3]
            glorot_aa: [b,6]
            poses: [b, 21*6]
            only_return_jts: bool

        Returns:
            output: np.array or torch.tensor, either joints [b, 22, 3] or markers [b, n_markers, 3]

        Raises:
            NotImplementedError rises if output_type is neither 'markers' nor 'joints'
        """
        nb = transl.shape[0]
        glorot_rotmat = RotConverter.cont2rotmat(glorot).reshape(nb,-1,3,3)
        poses_rotmat = RotConverter.cont2rotmat(poses.reshape(-1,6)).reshape(nb,-1,3,3)
        
        smplxout = self.bm(trans=transl, 
                           root_orient=glorot_rotmat, 
                           betas=betas, 
                           pose_body=poses_rotmat)
        jts = smplxout.Jtr
        if returntype == 'jts':
            return jts
        elif returntype == 'verts':
            return smplxout.v
        elif returntype == 'jts_body':
            return jts[:,:22]
        elif returntype == 'jts_and_ssm2_markers':
            markers = smplxout.v[:,self.marker_idx]
            return jts, markers
        elif returntype == 'jts_and_ssm2_markers_and_eyeverts':
            markers = smplxout.v[:,self.marker_idx]
            eyeverts = smplxout.v[:,[9929, 9448]]
            return jts, markers,eyeverts 
        elif returntype == 'jts_and_landmarks':
            ## 45 keypoints in total
            ## here we actually mix the joints and some landmarkers on the body surface to jts
            ## num_joints=55, following https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
            ## some additional landmarks: https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py
            ## here we only add [nose, reye, leye, rear, lear, ]
            jts_body = jts[:,:22]
            # jts_hand_joints = jts[:,25:] # 
            landmarkids = list(SMPLX_SURFACE_KEYPOINTS.values())
            landmarks_extra = smplxout.v[:,landmarkids] 
            keypoints = torch.cat([jts_body, landmarks_extra], dim=-2)
            return keypoints
        else:
            raise NotImplementedError



    def get_new_coordinate(self,
                betas: torch.Tensor,
                xb: torch.Tensor,
                coordinate_on_ground=False,
        ):
        """get a batch canonical coordinates for recursive use.
        Note that this function extends self.coord_extractor.get_new_coordinate, and assumes the first frame is the reference frame.
        Therefore, it is required to select the motion seed before using this function.

        Args:
            betas: the body shape, [b,10]
            xb: the compact vector of body params, (b, d)
            
        Returns:
            [new_rotmat, new_transl]: np.array or torch.tensor, in the shape of [b, 3, 3] and [b, 1, 3]

        Raises:
            None
        """
        assert betas.ndim==2, 'incorrect betas shape, should be [b,d]'
        transl = xb[...,:3]
        glorot_rotcont = xb[...,3:9]
        poses_rotcont = xb[...,9:]
        joints = self.forward_smplx(betas, transl, glorot_rotcont, poses_rotcont, returntype='jts')
        new_rotmat, new_transl = self.coord_extractor(joints)
        if coordinate_on_ground:
            new_transl[...,1] = 0 # only feasible when Y-up
            
        return new_rotmat, new_transl


    @torch.no_grad()
    def get_contact_features(self, contactidx, on_markers=True):
        """based on contact idx, 

        Args:
            contactidx: [b,] the mesh verts idx on smplx
            on_markers: if true, we only consider the markerset
        """
        assert on_markers, 'only support on_markers now...'

        marker_idx_tensor = torch.tensor(self.marker_idx)
        contact_labels_on_marker = []
        for cc in contactidx:
            cc = torch.tensor(cc)
            ccm = torch.isin(marker_idx_tensor, cc)
            contact_labels_on_marker.append(ccm)
        
        contact_labels_on_marker = torch.stack(contact_labels_on_marker)
        return contact_labels_on_marker.float()



    def calc_calibrate_offset(self,
                            betas: torch.Tensor,
                            body_pose_rotcont: torch.Tensor,
        ):
        """compensate the offset when transforming the smplx body and the body parameterss
        When performing the global transformation, first getting the body and then transforming will lead to a different result
        from first transforming the transl/global_orient and then get the body model. The reason is the global rotation is about
        the pelvis, whereas the global translation is about the kinematic tree root, which is not the body pelvis.

        Args:
            betas: [b,10]
            body_pose: the body pose in axis-angle, in shape (b, 63)
            
        Returns:
            delta_T: the compensation offset between root and body pelvis, in shape (b, 3)

        Raises:
            None
        """
        nb = body_pose_rotcont.shape[0]
        body_pose_rotmat = RotConverter.cont2rotmat(
            body_pose_rotcont.reshape(-1,6)).reshape(nb,-1,3,3)

        smplx_out = self.bm(pose_body=body_pose_rotmat, betas=betas)
        delta_T = smplx_out.Jtr[:,0,:] # we output all pelvis locations
        
        return delta_T

    @torch.no_grad()
    def update_transl_glorot(self,
                            transf_rotmat: torch.Tensor,
                            transf_transl: torch.Tensor,
                            betas: torch.Tensor,
                            xb: torch.Tensor,
                            fwd_transf: bool=False):
        """update the (global) body parameters when performing global transform
        When performing the global transformation, first getting the body and then transforming will lead to a different result
        from first transforming the transl/global_orient and then get the body model. The reason is the global rotation is about
        the pelvis, whereas the global translation is about the kinematic tree root, which is not the body pelvis.

        Args:
            transf_rotmat: torch.tensor or np.array in shape (b, 3, 3)
            transf_transl: torch.tensor or np.array in shape (b, 1, 3)
            betas: np.array or torch.tensor, in shape (10, ) or (1, 10)
            xb: torch.tensor or np.array in shape (b, 3+3+63), the smplx body parameters
            fwd_transf: if yes, apply (transf_rotmat, transf_transl) to xb; if false, transform xb to the coordinate (transf_rotmat, transf_transl)

        Returns:
            xb: the body parameters containing the updated transl and global_orient, (b, 3+3+63)

        Raises:
            None
        """
        
        delta_T = self.calc_calibrate_offset(betas,xb[:,9:135])
        transl = xb[:,:3]
        glorot = xb[:,3:9]
        
        if not fwd_transf:
            global_ori = RotConverter.cont2rotmat(glorot)
            global_ori_new = torch.einsum('bij,bjk->bik', transf_rotmat.permute(0,2,1), global_ori)
            glorot = RotConverter.rotmat2cont(global_ori_new)
            transl = torch.einsum('bij,bj->bi', transf_rotmat.permute(0,2,1), transl+delta_T-transf_transl[:,0])-delta_T
        else:
            global_ori = RotConverter.cont2rotmat(glorot)
            global_ori_new = torch.einsum('bij,bjk->bik', transf_rotmat, global_ori)
            glorot = RotConverter.rotmat2cont(global_ori_new)
            transl = torch.einsum('bij,bj->bi', transf_rotmat, transl+delta_T)-delta_T+transf_transl[:,0]

        xb[:,:3] = transl
        xb[:,3:9] = glorot

        return xb
    


    @torch.no_grad()
    def update_transl_glorot_seq(self,
                            transf_rotmat: torch.Tensor,
                            transf_transl: torch.Tensor,
                            betas: torch.Tensor,
                            xb_in: torch.Tensor,
                            fwd_transf: bool=False):
        """update the (global) body parameters when performing global transform
        When performing the global transformation, first getting the body and then transforming will lead to a different result
        from first transforming the transl/global_orient and then get the body model. The reason is the global rotation is about
        the pelvis, whereas the global translation is about the kinematic tree root, which is not the body pelvis.

        Args:
            transf_rotmat: torch.tensor or np.array in shape (b, 3, 3)
            transf_transl: torch.tensor or np.array in shape (b, 1, 3)
            betas: np.array or torch.tensor, in shape [b,t,d]
            xb: torch.tensor or np.array in shape (b, t, 3+3+63), the smplx body parameters
            fwd_transf: if yes, apply (transf_rotmat, transf_transl) to xb; if false, transform xb to the coordinate (transf_rotmat, transf_transl)

        Returns:
            xb: the body parameters containing the updated transl and global_orient, (b, t, 3+3+63)

        Raises:
            None
        """
        xb = xb_in.detach().clone()
        nb,nt = xb.shape[:2]
        delta_T_reshaped = self.calc_calibrate_offset(betas.reshape(nb*nt,-1),xb[:,:,9:135].reshape(nb*nt,-1))
        delta_T = delta_T_reshaped.reshape(nb,nt,-1)
        transl = xb[:,:,:3]
        glorot = xb[:,:,3:9]
        if not fwd_transf:
            global_ori = RotConverter.cont2rotmat(glorot)
            global_ori_new = torch.einsum('bij,btjk->btik', transf_rotmat.permute(0,2,1), global_ori)
            glorot_new = RotConverter.rotmat2cont(global_ori_new)
            transl_new = torch.einsum('bij,btj->bti', transf_rotmat.permute(0,2,1), transl+delta_T-transf_transl)-delta_T
        else:
            global_ori = RotConverter.cont2rotmat(glorot)
            global_ori_new = torch.einsum('bij,btjk->btik', transf_rotmat, global_ori)
            glorot_new = RotConverter.rotmat2cont(global_ori_new)
            transl_new = torch.einsum('bij,btj->bti', transf_rotmat, transl+delta_T)-delta_T+transf_transl

        xb[:,:,:3] = transl_new
        xb[:,:,3:9] = glorot_new

        return xb



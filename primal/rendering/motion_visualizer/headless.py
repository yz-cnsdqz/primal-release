import numpy as np
import cv2
import os

from .vis_utils import VideoSaver
from .visualizer import HeadlessVisualizer
from .camera import calculate_camera_position_from_trajectory
from gloss  import *


path_root = os.path.dirname(os.path.realpath(__file__) )
path_config = os.path.join(path_root,"./vis_configs/viewer_config.toml")
no_bg_floor_config = os.path.join(path_root,"./vis_configs/no_bg_floor.toml")



def motion_headless_rendering_kpts(
        poses, 
        trans, 
        betas, 
        kpts=None,
        kpts_contact=None,
        smpl_model_path='smpl_rs_suite/data/SMPLX_neutral_array_f32_slim.npz', \
        save_path = "temp.mp4", fps = 30, frame_size=(512, 512), \
        max_frame_limit=100000, gender='neutral', smpl_type='smplx',\
        camera_mode='lookat', cam_position=(-1,3,5), cam_lookat=(-1,1,0),
        camera_view='full_traj', fov_rad=0.7, cam_pitch_deg=20,
        vizer=None, show_floor=True
    ):
    
    print('\n\n')

    ############### define visualizer ###############
    video_saver = VideoSaver(save_path, vid_fps=fps, frame_size=frame_size)
    if vizer is None:
        vizer = HeadlessVisualizer(
            config_path=path_config, 
            smpl_model_path=smpl_model_path,
            fps=fps, 
            with_default_texture=contacts is None,
            frame_size=frame_size, 
            show_floor=show_floor
        )

    ############### load motion sequences ###############
    mesh = vizer.load_smpl_motion_sequence(poses, trans, betas=betas)   


    ############### define camera motions ###############
    if camera_mode=='follow':
        vizer.let_camera_following_this_mesh(mesh)
    elif camera_mode=='lookat':
        if camera_view=='full_traj':
            cam_position, cam_lookat = calculate_camera_position_from_trajectory(trans, fov=fov_rad, pitch=cam_pitch_deg)
        vizer.set_camera_positions(position=cam_position, lookat=cam_lookat)
    else:
        raise(f"{camera_mode} has not been implemented.")

    ############### rendering ###############
    anim_length = len(poses)
    rendered_frames = []
    
    if kpts is not None:
        if kpts_contact is None:
            kpts_color = np.ones_like(kpts) * (-1)
        else:
            kpts_color = np.zeros_like(kpts) + kpts_contact
            kpts_color[...,1:] = 0

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50) # org
    fontScale = 1 # fontScale
    color = (0, 0, 255) # Blue color in BGR
    thickness = 2 # Line thickness of 2 px
    
    # main loop
    for frame_ind in range(anim_length):
        # initialize the first frame and go to the next frame of animation when frame_ind>0.
        # current_frame_id start from 0
        if kpts is not None:
            current_time, current_frame_id = vizer.mesh_kpts_go_next_frame(
                mesh,kpts[frame_ind], kpts_color[frame_ind], fps=fps, advance=frame_ind>0)
            # raise NotImplementedError('kpts visualization has bugs due to gloss/smplrs updates')
        else:
            current_time, current_frame_id = vizer.mesh_go_next_frame(
                mesh, contacts=None, fps=fps, advance=frame_ind>0)

        assert current_frame_id == frame_ind, 'something wrong, it should have current_frame_id == frame_ind'

        # rendering the current frame step
        tex_numpy = vizer.get_rendering()
        rendered_frame = tex_numpy[:,:,:3][:,:,::-1].copy()
        rendered_frame = cv2.putText(
            rendered_frame, f"{current_frame_id}",org, font, 
            fontScale, color, thickness, cv2.LINE_AA
            ) # add frame txt to the image
        rendered_frames.append(rendered_frame)
        if save_path is not None:
            #cv2.imwrite(f'cache/{int(current_frame_id):06d}.jpg', rendered_frame)
            video_saver.write2video(rendered_frame)

        if current_frame_id>max_frame_limit:
            print(f'motion rendering break because the current frame index {current_frame_id} is out of limit {max_frame_limit}')
            break

    assert len(poses)==len(rendered_frames), print('motion rendered frame number not equal to pose number', poses.shape, len(rendered_frames))
    if save_path is not None:    
        video_saver.done()  

    return rendered_frames 










def motion_headless_rendering_with_kpts_head(
        poses, 
        trans, 
        betas, 
        kpts=None,
        kpts_contact=None,
        smpl_model_path='smpl_rs_suite/data/SMPLX_neutral_array_f32_slim.npz', \
        save_path = "temp.mp4", fps = 30, frame_size=(2048, 2048), \
        max_frame_limit=100000, gender='neutral', smpl_type='smplx',\
        camera_mode='lookat', cam_position=(-1,3,5), cam_lookat=(-1,1,0),
        camera_view='full_traj', fov_rad=0.7, cam_pitch_deg=20,
        vizer=None, show_floor=True
    ):

    if save_path is not None:
        video_saver = VideoSaver(save_path, vid_fps=fps, frame_size=frame_size)
    if vizer is None:
        vizer = Gloss_Visualizer(config_path=path_config, smpl_model_path=smpl_model_path, \
            smpl_type=smpl_type, gender=gender, fps=fps, with_default_texture=contacts is None,\
            frame_size=frame_size, headless=True, show_floor=show_floor)

    mesh = vizer.load_smpl_motion_sequence(poses, trans, betas=betas)   
    
    if camera_mode=='follow':
        vizer.let_camera_following_this_mesh(mesh)
    elif camera_mode=='lookat':
        if camera_view=='full_traj':
            cam_position, cam_lookat = calculate_camera_position_from_trajectory(trans, fov=fov_rad, pitch=cam_pitch_deg)
        vizer.set_camera_positions(position=cam_position, lookat=cam_lookat)
    else:
        raise(f"{camera_mode} has not been implemented.")
    
    anim_length = len(poses)
    rendered_frames = []
    
    if kpts is not None:
        if kpts_contact is None:
            kpts_color = np.ones_like(kpts) * (-1)
        else:
            kpts_color = np.zeros_like(kpts) + kpts_contact
            kpts_color[...,1:] = 0

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2


    for frame_ind in range(anim_length):
        # initialize the first frame and go to the next frame of animation when frame_ind>0.
        # current_frame_id start from 0
        if kpts is not None:
            current_time, current_frame_id = vizer.mesh_kpts_go_next_frame(
                mesh,kpts[frame_ind], kpts_color[frame_ind], fps=fps, advance=frame_ind>0)
        else:
            current_time, current_frame_id = vizer.mesh_go_next_frame(
                mesh, contacts=None, fps=fps, advance=frame_ind>0)

        
        assert current_frame_id == frame_ind, 'something wrong, current_frame_id != frame_ind'
        # rendering the current frame step
        tex_numpy = vizer.get_rendering()
        rendered_frame = tex_numpy[:,:,:3][:,:,::-1].copy()
        rendered_frame = cv2.putText(
            rendered_frame, f"{current_frame_id}",org, font, 
            fontScale, color, thickness, cv2.LINE_AA
            ) # add frame txt to the image
        rendered_frames.append(rendered_frame)
        if save_path is not None:
            #cv2.imwrite(f'cache/{int(current_frame_id):06d}.jpg', rendered_frame)
            video_saver.write2video(rendered_frame)

        if current_frame_id>max_frame_limit:
            print(f'motion rendering break because the current frame index {current_frame_id} is out of limit {max_frame_limit}')
            break

    assert len(poses)==len(rendered_frames), print('motion rendered frame number not equal to pose number', poses.shape, len(rendered_frames))
    if save_path is not None:    
        video_saver.done()  
    return rendered_frames 






def motion_headless_rendering_with_path(poses, trans, betas, wpath, contacts=None, \
                smpl_model_path='smpl_rs_suite/data/SMPLX_neutral_array_f32_slim.npz', \
                save_path = "temp.mp4", fps = 30, frame_size=(2048, 2048), \
                max_frame_limit=100000, gender='neutral', smpl_type='smplx',\
                camera_mode='lookat', cam_position=(-1,3,5), cam_lookat=(-1,1,0),
                camera_view='full_traj', fov_rad=0.7, cam_pitch_deg=20,
                vizer=None, show_floor=True):
    """_summary_

    Args:
        poses (_type_): _description_
        trans (_type_): _description_
        betas (_type_): _description_
        wpath (numpy.ndarray): a sequence of waypoints to represent the path, of shape (N, 3)
        contacts (_type_, optional): _description_. Defaults to None.
        fps (int, optional): _description_. Defaults to 30.
        frame_size (tuple, optional): _description_. Defaults to (2048, 2048).
        gender (str, optional): _description_. Defaults to 'neutral'.
        smpl_type (str, optional): _description_. Defaults to 'smplx'.
        cam_position (tuple, optional): _description_. Defaults to (-1,3,5).
        cam_lookat (tuple, optional): _description_. Defaults to (-1,1,0).
        camera_view (str, optional): _description_. Defaults to 'full_traj'.
        fov_rad (float, optional): _description_. Defaults to 0.7.
        cam_pitch_deg (int, optional): _description_. Defaults to 20.
        vizer (_type_, optional): _description_. Defaults to None.
        show_floor (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if save_path is not None:
        video_saver = VideoSaver(save_path, vid_fps=fps, frame_size=frame_size)
    if vizer is None:
        vizer = Gloss_Visualizer(config_path=path_config, smpl_model_path=smpl_model_path, \
            smpl_type=smpl_type, gender=gender, fps=fps, with_default_texture=contacts is None,\
            frame_size=frame_size, headless=True, show_floor=show_floor)

    mesh = vizer.load_smpl_motion_sequence(poses, trans, betas=betas)   
    
    if camera_mode=='follow':
        vizer.let_camera_following_this_mesh(mesh)
    elif camera_mode=='lookat':
        if camera_view=='full_traj':
            cam_position, cam_lookat = calculate_camera_position_from_trajectory(trans, fov=fov_rad, pitch=cam_pitch_deg)
        vizer.set_camera_positions(position=cam_position, lookat=cam_lookat)
    else:
        raise(f"{camera_mode} has not been implemented.")
    
    ## add walking paths here
    ## see example: https://gitlab.com/meshcapade/graphics/gloss/-/blob/main/bindings/gloss_py/examples/show_lines.py?ref_type=heads
    scene = vizer.viewer.get_scene()
    mesh_wpath = scene.get_or_spawn_renderable("path")
    mesh_wpath.insert(Verts(wpath.astype(np.float32))) 
    wpath_idx_from = np.arange(wpath.shape[0]-1)
    wpath_idx_to = wpath_idx_from + 1
    wpath_idx = np.stack([wpath_idx_from, wpath_idx_to], axis=1).astype(np.uint32)
    mesh_wpath.insert(Edges(wpath_idx))
    mesh_wpath.insert(VisLines(show_lines=True, line_width=10.0, zbuffer=False)) #Sometimes the lines represent something like a body skeleton which would be normally occluded by the rest of the body mesh. Setting zbuffer=False will render the lines as an overlay regardless if they are occluded by the zbuffer
    mesh_wpath.insert(VisPoints(show_points=True, point_size=10.0, zbuffer=False)) #Sometimes we would want to show the joints to and have the points in front. Points are given higher priority than lines due to visibility. Setting zbuffer=False will render the points farthest ahead regardless of occlusions



    anim_length = len(poses)
    rendered_frames = []
    for frame_ind in range(anim_length):
        # initialize the first frame and go to the next frame of animation when frame_ind>0.
        # current_frame_id start from 0
        current_time, current_frame_id = vizer.mesh_go_next_frame(mesh, contacts=contacts, fps=fps, advance=frame_ind>0)
        assert current_frame_id == frame_ind, 'something wrong, current_frame_id != frame_ind'
        # rendering the current frame step
        tex_numpy = vizer.get_rendering()
        rendered_frame = tex_numpy[:,:,:3][:,:,::-1].copy()
        rendered_frames.append(rendered_frame)
        if save_path is not None:
            #cv2.imwrite(f'cache/{int(current_frame_id):06d}.jpg', rendered_frame)
            video_saver.write2video(rendered_frame)

        if current_frame_id>max_frame_limit:
            print(f'motion rendering break because the current frame index {current_frame_id} is out of limit {max_frame_limit}')
            break

    assert len(poses)==len(rendered_frames), print('motion rendered frame number not equal to pose number', poses.shape, len(rendered_frames))
    if save_path is not None:    
        video_saver.done()  
    return rendered_frames 



def pose_headless_rendering(pose, tran, beta, vertex_colors=None, \
                smpl_model_path='smpl_rs_suite/data/SMPLX_neutral_array_f32_slim.npz', \
                save_path=None, frame_size=(2048, 2048), \
                gender='neutral', smpl_type='smplx',\
                camera_mode='lookat',cam_position=(0.0, 0.7, 3.8), cam_lookat=(0,1,0), cam_extrinsics = None,
                cam_focallen = (1600.0,1600.0), cam_center = (1024.0, 1024.0),
                vizer=None):
    if vizer is None:
        vizer = Gloss_Visualizer(config_path=path_config, smpl_model_path=smpl_model_path, \
            smpl_type=smpl_type, gender=gender, with_default_texture=vertex_colors is None,\
            frame_size=frame_size, headless=True)

    mesh = vizer.load_smpl_pose(pose, tran, beta=beta)   
    
    # set cam intrinsics
    cam = vizer.viewer.get_camera()
    cam.set_intrinsics(cam_focallen[0],cam_focallen[1],cam_center[0],cam_center[1])
    
    if camera_mode=='follow':
        vizer.let_camera_following_this_mesh(mesh)
    elif camera_mode=='lookat':
        vizer.set_camera_positions(position=cam_position, lookat=cam_lookat)
    elif camera_mode=='extrinsics':
        assert type(cam_extrinsics) == np.ndarray, f"You are using {camera_mode} mode, cam_extrinsics should be a numpy array of size (4,4)"
        assert cam_extrinsics.shape == (4,4), "cam_extrinsics should be a numpy array of size (4,4)"
        cam.set_extrinsics(cam_extrinsics)
    else:
        raise(f"{camera_mode} has not been implemented.")
    
    vizer.render_once(vertex_colors=vertex_colors, mesh=mesh)
    tex_numpy = vizer.get_rendering()
    rendered_image = tex_numpy.copy()
    rendered_image[:,:,:3] = tex_numpy[:,:,:3][:,:,::-1]

    if save_path is not None:
        cv2.imwrite(save_path, rendered_image)
    return rendered_image
   
    
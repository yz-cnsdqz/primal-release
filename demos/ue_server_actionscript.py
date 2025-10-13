"""
Backend server for scripted action sequences in Unreal Engine demo.

This script generates pre-scripted motion sequences using an action-conditioned
autoregressive diffusion model and streams them to Unreal Engine for rendering.
Unlike ue_server_interactive.py which responds to real-time user input, this
script executes a predefined motion script with specified actions, intensities,
directions, and durations.

Usage:
    python demos/ue_server_actionscript.py

Requirements:
    - Must be used with the corresponding Unreal Engine frontend (see UE project docs)
    - Configure UDP host IP and port in the script (lines 215-216)
    - Requires a trained ARDiffusionAction model checkpoint
    - Define motion script at lines 226-234 with action sequences

Motion Script Format:
    Each motion in the script is a dictionary with:
    - duration: Duration in seconds
    - action: Action type ('walk', 'run', 'jump', 'kick', 'punch')
    - intensity: Guidance weight for action conditioning (0.0-1.0+)
    - spatial_control: Enable/disable spatial control
    - movingdir: Movement direction in radians (0=left, π/2=forward)
    - movingspeed: Movement speed multiplier

The server runs in a producer-consumer architecture:
    - Producer thread: Generates motion frames following the script
    - Consumer thread: Sends SMPL poses to Unreal Engine via UDP
"""

import os, glob
import time
import json
from queue import Queue
import threading
from socket import *
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from primal.models.motion_diffuser import ARDiffusionAction

STATIC_ASSETS_DIR = "demos/ue_static"

###########################################################################
# setup class to Unreal livelink
###########################################################################

# Load and process joint names
joint_names = json.load(open(os.path.join(STATIC_ASSETS_DIR, "joint_names.json")))
joint_names = {int(k): v for k, v in joint_names.items()}


class SMPL_publisher:
    def __init__(self, host, port):
        self.addr = (host, port)
        self.UDPSock = socket(AF_INET, SOCK_DGRAM)
        self.set_zero_frame()

    def set_zero_frame(self):
        self.message = "A_mogen="
        for k,joint_name in joint_names.items():
            self.message += joint_name + ":(" + "{:.9f}".format(0)+ "," + "{:.9f}".format(-0) +  "," + "{:.9f}".format(-0) +  "," + "{:.9f}".format(-0) +  "," + "{:.9f}".format(0)+  "," + "{:.9f}".format(-0)+ "," + "{:.9f}".format(1)+ ")" + "|"
        self.message += "|" 
        
    def set_frame(self, poses, pelvis_position):
        poses = np.concatenate((poses, np.zeros((54 - poses.shape[0] + 1, 3))), axis=0)
        # convert to quaternions
        poses_quat = R.from_rotvec(poses).as_quat()
        joint_names_items = list(joint_names.items())
        self.message = "A_mogen="
        self.message += joint_names_items[0][1] + ":(" + "{:.9f}".format(pelvis_position[0])+ "," + "{:.9f}".format(-pelvis_position[1]) +  "," + "{:.9f}".format(pelvis_position[2]) +  \
            "," + "{:.9f}".format(-poses_quat[0,0]) +  "," + "{:.9f}".format(poses_quat[0,1])+  "," + "{:.9f}".format(-poses_quat[0,2])+ "," + "{:.9f}".format(poses_quat[0,3])+ ")" + "|"

        for k,joint_name in joint_names_items[1:]:
            self.message += joint_name + ":(" + "{:.9f}".format(0)+ "," + "{:.9f}".format(-0) +  "," + "{:.9f}".format(-0) +  \
            "," + "{:.9f}".format(-poses_quat[k,0]) +  "," + "{:.9f}".format(poses_quat[k,1])+  "," + "{:.9f}".format(-poses_quat[k,2])+ "," + "{:.9f}".format(poses_quat[k,3])+ ")" + "|"
        self.message += "|" 


    def send(self):
        self.UDPSock.sendto(self.message.encode(), self.addr)






###########################################################################
# utility functions for file IO
###########################################################################
def load_smpl_data(seqpath, n_frames=2, tgt_fps=30):
    data = np.load(seqpath, allow_pickle=True)

    #load shape params
    betas = data['shapeParameters']
    betas = betas[:16]
    if betas.shape[0] < 16:
        nbb = betas.shape[0]
        betas = np.pad(betas, (0,16-nbb), 'constant', constant_values=0)
    
    # load pose params
    if data['frameCount'] > 1:
        # assert data['frameRate'] >=15, "framerate should be >=15."
        if data['frameRate'] <30:
            #if fps<30, it is regarded as fast motion, and fps is set to 30
            realfps = 30
        else:
            # is fps>30, we will downsample the input sequence to 30fps
            realfps = data['frameRate']
        stride = int(realfps // tgt_fps)
        transl = data['bodyTranslation'][::stride]
        glorot_aa = data['bodyPose'][::stride,0]
        pose_body = data['bodyPose'][::stride,1:]
        pose_body = pose_body.reshape(-1,63)
        if n_frames is not None:
            transl = transl[:n_frames]
            glorot_aa = glorot_aa[:n_frames]
            pose_body = pose_body[:n_frames]
    elif data['frameCount'] == 1:
        transl = np.repeat(data['bodyTranslation'],2,axis=0)
        glorot_aa = np.repeat(data['bodyPose'],2,axis=0)[:,0]
        pose_body = np.repeat(data['bodyPose'],2,axis=0)[:,1:]
        pose_body = pose_body.reshape(-1,63)
    else:
        raise ValueError("data['frameCount'] should be >=1")
    
    xb = np.concatenate([transl, glorot_aa, pose_body], axis=-1)
    return betas, xb


###########################################################################
# main function for motion generation
###########################################################################
def backend_producer(queue, file, motion_script, **kwargs):
    # Prepare data and model
    betas1, xb1 = load_smpl_data(os.path.join(STATIC_ASSETS_DIR, 'jumping-jacks.smpl'))
    betas1 = torch.tensor(betas1).float().to(model.device)
    betas1 = betas1[None, None, :].repeat(1, xb1.shape[0], 1)
    xb1 = torch.tensor(xb1).float().to(model.device)[None, ...]
    
    batch = {}
    betas_zero = torch.zeros_like(betas1) # all zero shape parameters
    batch['betas'] = betas_zero
    batch['xb'] = xb1

    model.noise_scheduler.set_timesteps(kwargs["number_of_inference_steps"])
    

    ACTION_LABELS = ['jump', 'kick', 'punch', 'run', 'walk']

    switch_on_inertialization = True
    guidance_weight_mv = 50
    guidance_weight_facing = 25


    for motion in motion_script:
        current_action = motion["action"]
        guidance_weight_action = motion["intensity"]
        switch_on_control = motion["spatial_control"]
        ori_angle = motion["movingdir"]
        movingspeed = motion["movingspeed"]
        duration = motion["duration"]

        # orientation and speed
        ori = [np.cos(ori_angle), 0, np.sin(ori_angle)]
        ori = torch.tensor(ori)[None, None, ...].float().to(model.device)
        batch["ori"] = movingspeed * ori

        # setting the action label
        if current_action is not None:
            action_label = ACTION_LABELS.index(current_action)    
            batch["action_label"] = torch.tensor(action_label)[None,None].long().to(model.device) #[b=1,t=1]


        # generate motion for the duration amount of seconds
        starting_time = time.time()
        while time.time() - starting_time < duration:
            # Generate motion
            with torch.no_grad():
                _betas, xb_gen, _kpts, _outputlogs, _ = model.generate_perpetual_navigation(
                    batch, 
                    n_inference_steps = 28,
                    nt_max = 15,
                    snap_to_ground = True,
                    switch_on_inertialization = switch_on_inertialization,
                    switch_on_control = switch_on_control,
                    reproj_kpts = False,
                    guidance_weight_facing = guidance_weight_facing,
                    guidance_weight_action = guidance_weight_action,
                    guidance_weight_mv = guidance_weight_mv,
                )
            xb_gen = xb_gen[0]
            batch['xb'] = xb_gen[:, -2:, :]

            # Package frame data
            outputs = {
                "bodyPose": xb_gen[0, :-1, 3:3 + 3 * 22].reshape(-1, 22, 3).detach().cpu().numpy(),
                "bodyTranslation": xb_gen[0, :-1, :3].detach().cpu().numpy(),
                "nt": xb_gen.shape[1] - 1,
            }

            # Block if the buffer is full
            queue.put(outputs)



def frontend_consumer(queue):
    consumer_start_time = time.time()
    consumer_last_fps_print = time.time()
    consumer_frame_count = 0

    # while True:
    while not stop_event.is_set():
        data = queue.get()  # wait up to 1 second for data
        if data is None:
            # If we receive None, we assume the producer is done.
            print("Consumer received termination signal. Exiting loop.")
            break
        
        ######################### send current frame ##############################
        # note that "bodyTranslation" is the pelvis location!
        ###########################################################################
        nt = data['nt']
        for ii in range(nt):
            pub.set_frame(data["bodyPose"][ii], 100*data["bodyTranslation"][ii])
            pub.send()
            time.sleep(1./50) # in case of 1/30, large latency to unreal rendering.
            consumer_frame_count += 1

        current_time = time.time()
        if current_time - consumer_last_fps_print >= 5.0:
            consumer_total_elapsed = current_time - consumer_start_time
            consumer_avg_fps = consumer_frame_count / consumer_total_elapsed
            print(f"[Consumer] Avg FPS: {consumer_avg_fps:.2f}")
            consumer_last_fps_print = current_time



if __name__ == "__main__":

    ###########################################################################
    # CONFIGS
    ###########################################################################
    udp_listener_port = 3001
    udp_sender_host = "100.122.58.85"
    udp_sender_port = 2000
    frame_buffer_maxsize = 1

    hparams_file = 'logs/motion_diffuser_ar_action/runs/ours/tensorboard/version_0/hparams.yaml'
    model_cp_dir = 'logs/motion_diffuser_ar_action/runs/ours'


    ###########################################################################
    # MOTION SCRIPT TO EXECUTE AND STREAM TO UNREAL
    ###########################################################################
    motion_script = [
        {'duration': 4, 'action': 'walk', 'intensity': 1.0, 'spatial_control': False, 'movingdir': 0, 'movingspeed': 1},
        {'duration': 5, 'action': 'walk', 'intensity': 0.0, 'spatial_control': False, 'movingdir': 0, 'movingspeed': 1},
        {'duration': 5, 'action': 'punch', 'intensity': 1, 'spatial_control': False, 'movingdir': 0, 'movingspeed': 1},
        {'duration': 5, 'action': 'run', 'intensity': 0.3, 'spatial_control': True, 'movingdir': 0, 'movingspeed': 5},
        {'duration': 5, 'action': 'run', 'intensity': 0.3, 'spatial_control': True, 'movingdir': np.pi/2, 'movingspeed': 5},
        {'duration': 5, 'action': 'kick', 'intensity': 1.0, 'spatial_control': False, 'movingdir': np.pi/2, 'movingspeed': 5},
        {'duration': 5, 'action': 'punch', 'intensity': 0.3, 'spatial_control': True, 'movingdir': 0, 'movingspeed': 1.5},
    ]

    ###########################################################################
    # setup motion diffusion model
    ###########################################################################
    ckptpaths = sorted(
        glob.glob(os.path.join(model_cp_dir, 'tensorboard/version_0/checkpoints/*.ckpt')),
        key=os.path.getmtime
    )
    ## load the latest checkpoint
    ckptpath = ckptpaths[-1]
    print(f'loading model from {ckptpath}')
    model = ARDiffusionAction.load_from_checkpoint(
        ckptpath, 
        map_location='cuda:0',
        strict=False,
        hparams_file=hparams_file
    )
    model.load_ema_parameters()
    model.eval()
    model.freeze()


    ###########################################################################
    # inititalize consumer, producer and utility threads
    ###########################################################################


    # create SMPL publisher
    pub = SMPL_publisher(host=udp_sender_host, port=udp_sender_port)

    # create buffer queue
    buffer_queue = Queue(maxsize=frame_buffer_maxsize)
    stop_event = threading.Event()

    # Set up the producer thread
    producer_thread = threading.Thread(
        target = backend_producer, 
        args = (buffer_queue, os.path.join(STATIC_ASSETS_DIR,'jumping-jacks.smpl'), motion_script),
        kwargs = {
                "snap_to_ground": True,
                "reproj_kpts": False,
                "switch_on_control": True,
                "number_of_inference_steps": 50,
                "guidance_weight_mv": 1.0,
                "guidance_weight_facing": 50,
                "switch_on_inertialization": True
        }
    )
    # Set up the consumer thread
    consumer_thread = threading.Thread(target=frontend_consumer, args=(buffer_queue,))

    # Start both threads
    producer_thread.start()
    consumer_thread.start()
        
    try:
        # Wait for both to finish
        producer_thread.join()
        consumer_thread.join()

    except KeyboardInterrupt:
        # On Ctrl+C, signal threads to stop
        print("KeyboardInterrupt received. Stopping threads...")
        stop_event.set()
        
        # Put None in the queue in case consumer is blocked on q.get()
        buffer_queue.put(None)

        # Attempt a clean shutdown
        producer_thread.join()
        consumer_thread.join()

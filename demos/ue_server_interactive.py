"""
Backend server for interactive Unreal Engine avatar demo.

This script provides the motion generation backend for a real-time interactive
avatar demo in Unreal Engine. It receives user control inputs via UDP, generates
SMPL body motions using an autoregressive diffusion model, and streams the results
back to Unreal Engine for rendering.

Usage:
    python demos/ue_server_interactive.py

Requirements:
    - Must be used with the corresponding Unreal Engine frontend (see UE project docs)
    - Configure UDP ports and host IP addresses in the script (lines 497-499)
    - Requires a trained ARDiffusionSpatial model checkpoint

The server runs in a producer-consumer architecture:
    - UDP listener thread: Receives control inputs (WASD, gamepad, mouse clicks)
    - Producer thread: Generates motion frames using the diffusion model
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

from primal.models.motion_diffuser import ARDiffusionSpatial


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



def transform_poke_vector_3d(global_poke, orientation):
    """
    Returns a 3D vector [left, up, forward] from a global poke [F, R, U],
    given a character orientation in radians (0=left, π/2=forward, etc.).
    Orientation is not updated—this just transforms the poke vector.
    """
    F, R, U = global_poke

    angle = np.pi - orientation
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    left = cos_a*F - sin_a*R
    forward = sin_a*F + cos_a*R

    return np.array([left, U, forward])


def transform_poke_vector_planar_simple(global_poke):
    """
    Given a global "poke" vector (in Unreal's [forward, right, up] coords)
        calculate the orientation that points exactly in the poke's direction
        (0 = left, π/2 = forward, π = right, -π/2 = backward)
    Args:
        global_poke (array_like): [F, R, U] (we ignore U).
    Returns:
        new_orientation = arctan2(F, -R).
    """
    F = global_poke[0]  # global forward
    R = global_poke[1]  # global right
    
    # Orientation:
    # By your convention: 0 rad = left, π/2 = forward, π = right, -π/2 = backward.
    # A global vector F>0 means "more forward," R<0 means "more left," etc.
    # We use arctan2(forward, -right) so that:
    #   - if R=0, F>0 => orientation=π/2
    #   - if F=0, R<0 => orientation=0
    #   - if F=0, R>0 => orientation=π
    new_orientation = np.arctan2(F, -R)

    # print(f"Global poke: {global_poke}, orientation={new_orientation}")
    return new_orientation


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
# get controls from unreal
###########################################################################
def parse_control_data(inputs_stream):
    control_update = {}

    # print(inputs_stream)
    inputs_stream = inputs_stream.replace("(", "").replace(")", "")
    inputs_stream = inputs_stream.split("|")
    inputs_stream = [x for x in inputs_stream if x]  # Remove empty strings
    inputs_dict = {x.split(":")[0]: x.split(":")[1] for x in inputs_stream}
    # print(f"Parsed inputs: {inputs_dict}")

    # parse poking / pulling
    if inputs_dict["POKE_MODE"] == "1":
        poke_joint = int(inputs_dict["BoneIdx"])
        poke_force = float(inputs_dict["Force"])
        poke_vector = np.array([float(x) for x in inputs_dict["Vector"].split(",")])
        if inputs_dict["X"] == "1":
            poke_force = 25 * poke_force
        if poke_force != 0 and poke_joint != 0:
            # print(f"Poke joint: {poke_joint}, vector: {poke_vector}")
            poke_input = (poke_joint, -1 * poke_force * poke_vector)
            control_update["poke"] = poke_input
            control_update["pull"] = None
    else:
        pull_direction_global = np.array([float(x) for x in inputs_dict["COOKIE_T"].split(",")])
        control_update["pull"] = 1 * pull_direction_global / 500 # invert and scale down
        control_update["poke"] = None
        print(f"Pull direction: {pull_direction_global}")


    # parse movement from keyboard
    if inputs_dict["K_W"] == "1":
        control_update["forward"] = True
    if inputs_dict["K_S"] == "1":
        control_update["backward"] = True
    if inputs_dict["K_A"] == "1":
        control_update["left"] = True
    if inputs_dict["K_D"] == "1":
        control_update["right"] = True
    if inputs_dict["K_Space"] == "1":
        control_update["jump"] = True
    if inputs_dict["K_Q"] == "1":
        control_update["run"] = True

    # parse controller thumbsticks
    if inputs_dict["TS_L"] != "0.000000000,0.000000000":
        ts_l = np.array([float(x) for x in inputs_dict["TS_L"].split(",")])
        control_update["ts_l"] = ts_l
    if inputs_dict["TS_R"] != "0.000000000,0.000000000":
        ts_r = np.array([float(x) for x in inputs_dict["TS_R"].split(",")])
        control_update["ts_r"] = ts_r

    if inputs_dict["K_P"] == "1":
        control_update["reset"] = True
        print("P key pressed - this should restart (reset) the backend server")

    # parse movement from controller
    if inputs_dict["D_U"] == "1":
        control_update["forward"] = True
    if inputs_dict["D_D"] == "1":
        control_update["backward"] = True
    if inputs_dict["D_L"] == "1":
        control_update["left"] = True
    if inputs_dict["D_R"] == "1":
        control_update["right"] = True
    if inputs_dict["A"] == "1":
        control_update["jump"] = True
    if inputs_dict["B"] == "1":
        control_update["run"] = True

    return control_update



def udp_listener(port):
    udp_sock = socket(AF_INET, SOCK_DGRAM)
    udp_sock.bind(("0.0.0.0", port))
    udp_sock.setblocking(False)

    while not stop_event.is_set():
        try:
            data, _ = udp_sock.recvfrom(1024)
            control_update = parse_control_data(data.decode())

            with control_inputs_lock:
                control_inputs.update(control_update)

        except BlockingIOError:
            time.sleep(0.01)  # Avoid busy-waiting

    udp_sock.close()


###########################################################################
# main function for motion generation
###########################################################################
def backend_producer(queue, file, **kwargs):
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

    tt = 0
    ori_angle = np.pi / 2
    move_angle = ori_angle
    target_distance_speed = 0.1
    use_vel_to_jump = False

    poking_frames_remaining = 0
    jumping_frames_remaining = 0
    orientation_lock_frames = 0

    while not stop_event.is_set():
        with control_inputs_lock:
            if control_inputs.get("reset", False):
                print("Reset triggered. Stopping producer...")
                queue.put(None)  # Signal consumer to stop
                stop_event.set()
                control_inputs["reset"] = False
                break

            poking_vector = None
            poking_joint = None

            # Initialize speed decay rate
            speed_decay = 0.2
            min_speed = 0.02
            moving = False  # Flag to check if any movement key is pressed

            # Process movement inputs
            if control_inputs["forward"] or control_inputs["left"] or control_inputs["right"] or control_inputs["backward"] or control_inputs["ts_l"] is not None:
                target_distance_speed = 0.5
                moving = True

            if control_inputs["forward"] and control_inputs["left"]:
                ori_angle = np.pi / 4
                move_angle = ori_angle
            elif control_inputs["forward"] and control_inputs["right"]:
                ori_angle = 3 * np.pi / 4
                move_angle = ori_angle
            elif control_inputs["backward"] and control_inputs["left"]:
                ori_angle = -1 * np.pi / 4
                move_angle = ori_angle
            elif control_inputs["backward"] and control_inputs["right"]:
                ori_angle = -3 * np.pi / 4
                move_angle = ori_angle
            elif control_inputs["forward"]:
                ori_angle = np.pi / 2
                move_angle = ori_angle
            elif control_inputs["backward"]:
                ori_angle = -np.pi / 2
                move_angle = ori_angle
            elif control_inputs["left"]:
                ori_angle = 0
                move_angle = ori_angle
            elif control_inputs["right"]:
                ori_angle = np.pi
                move_angle = ori_angle

            # controller thumbstick input
            elif control_inputs["ts_l"] is not None:
                ts_l = control_inputs["ts_l"]
                ts_l[1] = -ts_l[1]
                if control_inputs["ts_r"] is not None:
                    ts_r = control_inputs["ts_r"]
                    # ts_l controls movement, ts_r controls orientation
                    move_angle = np.arctan2(ts_r[1], ts_r[0])
                    ori_angle = np.arctan2(ts_l[1], ts_l[0])
                else:
                    # ts_l controls both movement and orientation
                    ori_angle = np.arctan2(ts_l[1], ts_l[0])
                    move_angle = ori_angle
                # print(f"TS input: {ts_l}, orientation: {ori_angle}, movement angle: {move_angle}")
            elif control_inputs["ts_r"] is not None:
                ts_r = control_inputs["ts_r"]
                # ts_r controls orientation
                ori_angle = np.arctan2(ts_r[1], ts_r[0])
                move_angle = ori_angle
                # print(f"TS input 2: {ts_r}, orientation: {ori_angle}, movement angle: {move_angle}")

            # Handle jumping
            if control_inputs["jump"] and jumping_frames_remaining == 0:
                jumping_frames_remaining = 4
                use_vel_to_jump = True
                switch_on_snapping=False
            else:
                use_vel_to_jump = False
                switch_on_snapping=True

            if jumping_frames_remaining > 0:
                jumping_frames_remaining -= 1


            # Adjust speed for running
            if control_inputs["run"]:
                target_distance_speed = 2.5
                moving = True

            # Process poking or pulling input
            if control_inputs["poke"] is not None and tt > 0 and poking_frames_remaining == 0:
                poking_joint, poking_vector = control_inputs["poke"]
                if poking_joint > 21:
                    poking_joint = 0
                poking_vector = transform_poke_vector_3d(poking_vector, ori_angle)

                poking_vector = poking_vector / 2000.0 # scale down poking a lot
                poking_frames_remaining = 6
                print(f"Poked joint {poking_joint} with impulse {poking_vector}")

            elif control_inputs["pull"] is not None and tt > 0:
                pull_force = control_inputs["pull"]
                new_ori_angle = transform_poke_vector_planar_simple(pull_force)
                ori_angle = new_ori_angle
                move_angle = ori_angle
                target_distance_speed = 1
                guidance_weight_action = 1.5
                moving = True
                # print(f"Pulled towards orientation {new_ori_angle}")

            # print(control_inputs)
            # clear control inputs
            control_inputs["poke"] = None
            control_inputs["pull"] = None
            control_inputs["forward"] = False
            control_inputs["backward"] = False
            control_inputs["left"] = False
            control_inputs["right"] = False
            control_inputs["ts_l"] = None
            control_inputs["ts_r"] = None
            control_inputs["jump"] = False
            control_inputs["run"] = False

            ### end of "with control_inputs_lock" block

        # Disable control after poking
        if poking_frames_remaining > 0:
            switch_on_control = False
            switch_on_inertialization = False
            switch_on_snapping = True
            guidance_weight_action = 0
            guidance_weight_facing = 0
            poking_frames_remaining -= 1
        else:
            switch_on_control = True
            switch_on_inertialization = True
            guidance_weight_action = 1
            guidance_weight_facing = 50


        # Apply slowing down when no movement input is active
        if not moving:
            target_distance_speed = max(target_distance_speed - speed_decay, min_speed)

        # When standing still, 1/2 chance to change orientation by 30-45 degrees in either direction
        if target_distance_speed == 0.02 and orientation_lock_frames == 0:
            if np.random.rand() < 0.5:
                delta = np.deg2rad(np.random.uniform(30, 45)) * np.random.choice([-1, 1])
                ori_angle = (ori_angle + delta + np.pi) % (2 * np.pi) - np.pi
                move_angle = ori_angle
        if orientation_lock_frames > 0:
            orientation_lock_frames -= 1


        movement_vector = torch.tensor([np.cos(move_angle), 0, np.sin(move_angle)])[None, None, ...].float().to(model.device)
        ori = [np.cos(ori_angle), 0, np.sin(ori_angle)]
        ori = torch.tensor(ori)[None, None, ...].float().to(model.device)
        batch["ori"] = movement_vector

        _betas, xb_gen, _kpts,  = model.generate_perpetual_navigation_ue(
            batch, 
            n_inference_steps = 40,
            nt_max = 15,
            snap_to_ground = switch_on_snapping,
            switch_on_inertialization = switch_on_inertialization,
            switch_on_control = switch_on_control,
            guidance_weight_facing = guidance_weight_facing,
            guidance_weight_action = guidance_weight_action,
            target_distance_speed = target_distance_speed,
            poking_vector = poking_vector,
            poking_joint = poking_joint,
            movement_vector = ori,
            use_vel_to_jump = use_vel_to_jump,
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
        queue.put(outputs)  # Blocks until space is available

        tt += 1



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

    
# Function to monitor the queue size
def monitor_queue_size(buffer_queue, stop_event):
    while not stop_event.is_set():
        queue_size = buffer_queue.qsize()  # Get the current size of the queue
        print(f"Current queue size: {queue_size}")
        time.sleep(5)  # Monitor every 10 second (adjust as needed)



if __name__ == "__main__":

    ###########################################################################
    # CONFIGS
    ###########################################################################
    udp_listener_port = 3001
    udp_sender_host = "100.122.58.85"
    udp_sender_port = 2000
    frame_buffer_maxsize = 1
    
    # need to specify the model checkpoint path
    model_cp_dir = 'logs/motion_diffuser_ar_spatial/runs/ours'
    
    # when deploying the model to different machines. Some settings might be adapted.
    # create a new file with new settings, and specify its path.
    hparams_file = 'logs/motion_diffuser_ar_spatial/runs/ours/tensorboard/version_0/hparams.yaml'
    

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
    model = ARDiffusionSpatial.load_from_checkpoint(
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

    control_inputs = {
        "forward": False,
        "backward": False,
        "left": False,
        "right": False,
        "ts_l": None,
        "ts_r": None,
        "poke": None,
        "pull": None,
        "jump": False,
        "run": False,
        "reset": False,
    }
    control_inputs_lock = threading.Lock()


    # create SMPL publisher
    pub = SMPL_publisher(host=udp_sender_host, port=udp_sender_port)

    while True:
        control_inputs["reset"] = False

        # create buffer queue
        buffer_queue = Queue(maxsize=frame_buffer_maxsize)
        stop_event = threading.Event()

        # Set up the producer thread
        producer_thread = threading.Thread(
            target = backend_producer, 
            args = (buffer_queue, os.path.join(STATIC_ASSETS_DIR, 'jumping-jacks.smpl')),
            kwargs = {
                "snap_to_ground": True,
                "reproj_kpts": True,
                "switch_on_control": True,
                "number_of_inference_steps": 50,
                "guidance_weight_mv": 1.0,
                "guidance_weight_facing": 50,
                "switch_on_inertialization": True
            }
        )
        
        # Set up the consumer thread
        consumer_thread = threading.Thread(target=frontend_consumer, args=(buffer_queue,))

        # Monitoring thread to track the queue size
        monitor_thread = threading.Thread(target=monitor_queue_size, args=(buffer_queue, stop_event))

        # control stream thread
        control_stream_thread = threading.Thread(target=udp_listener, args=(udp_listener_port,), daemon=False)

        # Start both threads
        producer_thread.start()
        consumer_thread.start()
        monitor_thread.start()
        control_stream_thread.start()
        
        try:
            # Wait for both to finish
            producer_thread.join()
            consumer_thread.join()
            control_stream_thread.join()
            monitor_thread.join()

        except KeyboardInterrupt:
            # On Ctrl+C, signal threads to stop
            print("KeyboardInterrupt received. Stopping threads...")
            stop_event.set()
        
            # Put None in the queue in case consumer is blocked on q.get()
            buffer_queue.put(None)

            # Attempt a clean shutdown
            producer_thread.join()
            consumer_thread.join()
            monitor_thread.join()
            control_stream_thread.join()

import os
import subprocess
import pickle
import sys


from primal.rendering.motion_visualizer.headless import motion_headless_rendering_kpts
from primal.rendering.motion_visualizer.visualizer import HeadlessVisualizer


MODEL_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH", "/home/yan/model-registry")
vizer_path = os.path.join(os.path.dirname(__file__), 'motion_visualizer')
vis_smpl_model_path=os.path.join(MODEL_REGISTRY_PATH, "models/SMPLX/SMPLX_neutral_array_f32_slim.npz")
FRAME_SIZE=(512,512)
N_SEQS=50


vizer = HeadlessVisualizer(
    config_path=os.path.join(vizer_path,"vis_configs/viewer_config.toml"), 
    smpl_model_path=vis_smpl_model_path,
    smpl_type='smplx', 
    gender='neutral', 
    fps=30, 
    frame_size=FRAME_SIZE, 
    floor_type='default'
)


def renderfile(seqfile, renderkpts=False):

    with open(seqfile, 'rb') as f:
        data_all0 = pickle.load(f)
    
    outputdir = seqfile+'_renders'
    os.makedirs(outputdir, exist_ok=True)


    if type(data_all0) is dict:
        data_all = []
        n_samples = data_all0['betas'].shape[0]
        for b in range(n_samples):
            data_ = {}
            for k in data_all0.keys():
                data_[k] = data_all0[k][b:b+1]
            data_all.append(data_)
    else:
        data_all = data_all0


    if 'MotionRealism' in seqfile:
        N_FRAMES = 240
    elif 'Principled' in seqfile:
        N_FRAMES = 90
    elif 'ARDiffusionSpatial' in seqfile:
        N_FRAMES = 60
    elif 'ARDiffusionAction' in seqfile:
        N_FRAMES = 90
    else:
        # raise ValueError(f'Unknown setting: {seqfile}')
        N_FRAMES = 240


    for idx, data in enumerate(data_all):
        if idx >= N_SEQS:
            break

        if 'kpts' not in data.keys():
            betas, xb = data['betas'], data["xb"]
            kpts=None
        else:
            betas, xb, kpts = data['betas'], data["xb"], data['kpts']

        b = 0
        betas_ = betas[b,0,:10]
        transl = xb[b,:N_FRAMES,:3]
        poses = xb[b,:N_FRAMES,3:]
        kpts = kpts[b] if kpts is not None else None
        if not renderkpts:
            kpts = None

        if kpts is not None:
            kpts_locs = kpts[...,:3]
            kpts_color = kpts[...,-1:]

        seqfilename = os.path.join(outputdir,f'res_{idx:05d}.mp4')
        motion_headless_rendering_kpts(poses=poses, trans=transl, betas=betas_, 
                                    kpts=kpts_locs if kpts is not None else None,
                                    kpts_contact=kpts_color if kpts is not None else None,
                                    frame_size=FRAME_SIZE,
                                    save_path=seqfilename,
                                    camera_mode='follow', #'lookat',
                                    camera_view='full_traj',
                                    vizer=vizer,
                                    fov_rad=0.7 
                                    )
        subprocess.call(f'ffmpeg -y -i {seqfilename} -c:v libx264 -c:a aac -vf format=yuv420p {seqfilename}.mp4', shell=True)




if __name__=='__main__':
    import pickle
    import sys
    
    # visualze the keypoints or velocities? 
    renderkpts = sys.argv[2].lower() == 'true'
    pklfilename = sys.argv[1]

    renderfile(pklfilename, renderkpts)

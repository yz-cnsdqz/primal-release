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
        data = pickle.load(f)

    if 'kpts' not in data.keys():
        betas, xb = data['betas'], data["xb"]
        kpts=None
    else:
        betas, xb, kpts = data['betas'], data["xb"], data['kpts']

    
    # for b in range(betas.shape[0]):
    b = 0
    betas_ = betas[b,0,:10]
    transl = xb[b,:,:3]
    poses = xb[b,:,3:]
    kpts = kpts[b] if kpts is not None else None
    if not renderkpts:
        kpts = None

    seqfilename = seqfile+'.mp4'
    motion_headless_rendering_kpts(poses=poses, trans=transl, betas=betas_, 
                                   kpts=kpts[...,:-1] if kpts is not None else None,
                                   kpts_contact=kpts[...,-1:] if kpts is not None else None,
                                frame_size=FRAME_SIZE,
                                save_path=seqfilename,
                                camera_mode='follow', #'lookat',
                                camera_view='full_traj',
                                vizer=vizer,
                                fov_rad=0.7
                                )
    subprocess.call(f'ffmpeg -y -i {seqfilename} -c:v libx264 -c:a aac -vf format=yuv420p {seqfilename}.mp4', shell=True)




if __name__=='__main__':
    
    renderkpts = sys.argv[2].lower() == 'true'
    renderfile(sys.argv[1], renderkpts)

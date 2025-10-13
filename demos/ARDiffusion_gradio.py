#!/usr/bin/env python3
"""
Autoregressive Diffusion Motion Generation Gradio Interface

This module provides a Gradio web interface for generating human motion sequences
using autoregressive diffusion models. It allows users to upload motion files,
set generation parameters, and download results.
"""

import os
import glob
import random
from os.path import join as opj
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import numpy as np
import gradio as gr
import tyro
from omegaconf import OmegaConf

from primal.models.motion_diffuser import ARDiffusion
from primal.utils.data_io import load_data, save_data_pkl, save_data_smplcodec




def main():
    """Main application entry point."""
    # Parse arguments using tyro
    args = tyro.cli(LaunchArgs, description="Launch Gradio interface for autoregressive motion generation")
    
    device = Constants.get_device()
    
    # Initialize components
    model_manager = ModelManager(args.model_dir, device)
    model = model_manager.load_model()
    
    motion_generator = MotionGenerator(model, args)
    gradio_interface = GradioInterface(motion_generator)
    
    # Launch interface
    interface = gradio_interface.create_interface()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )




@dataclass
class LaunchArgs:
    """Configuration for the Gradio interface."""
    model_dir: str
    """Path to the model directory containing checkpoints"""
    
    host: str = "127.0.0.1"
    """Server host address"""
    
    port: int = 8000
    """Server port number"""
    
    share: bool = False
    """Whether to create a public shareable link"""
    
    debug: bool = True
    """Enable debug mode"""
    
    output_dir: str = "outputs/gradio_ARDiffusion"
    """Output directory for generated motions"""
    
    data_config_path: str = "primal/configs/data/amass_smplx.yaml"
    """Path to data configuration file"""
    
    subsets: Optional[list[str]] = None
    """Dataset subsets to use (defaults to ['SFU'])"""
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.subsets is None:
            self.subsets = ['SFU']
        # Create the output directory
        os.makedirs(self.output_dir, exist_ok=True)


class Constants:
    """Model-specific constants that should not be configurable."""
    
    ACTION_LIST = ['left_kick', 'right_kick', 'run_forward', 'flip_back', 'roll_forward', 'none']
    CHECKPOINT_PATTERN = 'tensorboard/version_0/checkpoints/*=29999-*.ckpt'
    HPARAMS_PATH = 'tensorboard/version_0/hparams.yaml'
    
    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')


class ModelManager:
    """Manages model loading and initialization."""
    
    def __init__(self, model_dir: str, device: torch.device):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        
    def load_model(self) -> ARDiffusion:
        """Load and initialize the motion diffusion model."""
        ckpt_paths = self._find_checkpoint_paths()
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints found in {self.model_dir}")
            
        latest_ckpt = ckpt_paths[-1]
        print(f'Loading model from {latest_ckpt}')
        
        hparams_file = opj(self.model_dir, Constants.HPARAMS_PATH)
        
        self.model = ARDiffusion.load_from_checkpoint(
            latest_ckpt,
            map_location=self.device,
            strict=False,
            hparams_file=hparams_file
        )
        
        self.model.load_ema_parameters()
        self.model.eval()
        self.model.freeze()
        
        return self.model
        
    def _find_checkpoint_paths(self) -> list:
        """Find all checkpoint files in the model directory."""
        pattern = opj(self.model_dir, Constants.CHECKPOINT_PATTERN)
        return sorted(glob.glob(pattern), key=os.path.getmtime)


class DataManager:
    """Manages data loading and configuration."""
    
    def __init__(self, args: LaunchArgs):
        self.args = args
        self.data_config = self._load_data_config()
        
    def _load_data_config(self) -> OmegaConf:
        """Load data configuration."""
        data_config = OmegaConf.load(self.args.data_config_path)
        data_config.subsets = self.args.subsets
        data_config.batch_size = 1
        data_config.seq_len = 2  # Load entire sequence one by one
        return data_config
        
    def get_random_file(self) -> str:
        """Get a random file from the dataset."""
        subset = random.choice(self.data_config.subsets)
        seq_files = glob.glob(os.path.join(self.data_config.path, subset, '*/*.npz'))
        return random.choice(seq_files)


class MotionGenerator:
    """Handles motion generation logic."""
    
    def __init__(self, model: ARDiffusion, args: LaunchArgs):
        self.model = model
        self.args = args
        self.data_manager = DataManager(args)
        os.makedirs(args.output_dir, exist_ok=True)
    
    def _prepare_batch_data(self, file_path: Optional[str]) -> Dict[str, torch.Tensor]:
        """Prepare batch data for motion generation."""
        if file_path is None:
            file_path = self.data_manager.get_random_file()
            
        betas, xb = load_data(file_path)
        
        # Convert to tensors and move to device
        betas_tensor = torch.tensor(betas).float().to(self.model.device)
        betas_tensor = betas_tensor[None, None, :].repeat(1, xb.shape[0], 1)
        xb_tensor = torch.tensor(xb).float().to(self.model.device)[None, ...]
        
        return {
            'betas': betas_tensor,
            'xb': xb_tensor
        }
    
    def _compute_orientation(self, angle_deg: float, speed: float) -> torch.Tensor:
        """Compute movement orientation vector from angle and speed."""
        angle_rad = np.deg2rad(angle_deg)
        # Note: Y is pointing up
        ori = [np.cos(angle_rad), 0, np.sin(angle_rad)]
        ori_tensor = torch.tensor(ori)[None, None, ...]  # [b=1, t=1, 3]
        return speed * ori_tensor.float().to(self.model.device)
    
    def _save_results(self, betas: torch.Tensor, xb_gen: torch.Tensor, 
                     kpts: torch.Tensor) -> Tuple[str, str]:
        """Save generated motion results to files."""
        idx = random.randint(0, int(1e8))
        
        # Save PKL file
        pkl_path = save_data_pkl(self.args.output_dir, idx, betas, xb_gen, kpts)
        
        # Render video
        cmd = f"python primal/rendering/render_gradio.py {pkl_path} False"
        mp4_path = pkl_path + '.mp4.mp4'
        os.system(cmd)
        
        # Save NPZ file
        # npz_path = save_data_npz(self.args.output_dir, idx, betas, xb_gen, self.model.fps)
        npz_path = save_data_smplcodec(self.args.output_dir, idx, betas, xb_gen, self.model.fps)
        
        return npz_path, mp4_path
    
    def generate(
        self,
        file: Optional[str],
        snap_to_ground: bool,
        use_vel_perturbation: bool,
        reproj_kpts: bool,
        switch_on_inertialization: bool,
        switch_on_control: bool,
        action: str,
        ori_angle_deg: float,
        speed: float,
        number_of_frames: int,
        number_of_inference_steps: int,
        guidance_weight_mv: float,
        guidance_weight_facing: float,
    ) -> Tuple[str, str, str]:
        """Generate motion sequence with specified parameters."""
        
        # Prepare input data
        batch = self._prepare_batch_data(file)
        
        # Set movement orientation
        batch["ori"] = self._compute_orientation(ori_angle_deg, speed)
        
        # Generate motion
        betas, xb_gen, kpts, output_logs = self.model.generate_perpetual_navigation(
            batch,
            n_inference_steps=number_of_inference_steps,
            nt_max=number_of_frames,
            snap_to_ground=snap_to_ground,
            use_vel_perburbation=use_vel_perturbation,
            switch_on_control=switch_on_control,
            switch_on_inertialization=switch_on_inertialization,
            reproj_kpts=reproj_kpts,
            guidance_weight_mv=guidance_weight_mv,
            guidance_weight_facing=guidance_weight_facing,
            perform_principled_action=action,
        )
        
        xb_gen = xb_gen[0]
        kpts = kpts[0]
        

            
        # Save results
        npz_path, mp4_path = self._save_results(betas, xb_gen, kpts)
        
        return npz_path, mp4_path, output_logs


class GradioInterface:
    """Manages the Gradio web interface."""
    
    ARTICLE = """
This interface generates perpetual human motions using the PRIMAL base model. The system can create realistic continuations from initial poses with various control options.

### Input Parameters:

1. **Input File**: Upload an AMASS .npz or .smpl file containing initial motion data. Only the first ≤2 frames are used as conditioning. If no file is provided, a random pose with velocities will be sampled from the AMASS dataset.

2. **Motion Processing Options**:
   - **Snap to Ground**: Forces the generated motion to maintain contact with the ground plane
   - **Add Random External Perturbation**: Introduces stochastic variations to create more natural motion dynamics
   - **Reproject Keypoints to SMPLX**: Projects predicted joint positions back to the SMPLX body model for improved anatomical consistency
   - **Switch on Inertialization**: Enables smooth motion transitions and reduces artifacts

3. **Control Parameters**:
   - **Switch on Control**: Enables directional and velocity guidance for the generated motion
   - **Moving Orientation**: Direction of movement in degrees (0° = rightward, 90° = outward/forward, 180° = leftward)
   - **Moving Speed**: Target velocity in meters per second (0-10 m/s range)
   - **Action**: Select from predefined actions (left_kick, right_kick, run_forward, flip_back, roll_forward) to generate specific motion behaviors

4. **Generation Settings**:
   - **Number of Frames**: Length of generated sequence (15-3000 frames at 30fps)
   - **Denoising Steps**: Number of diffusion inference steps (higher = better quality, slower generation)
   - **Guidance Weights**: Control strength for movement velocity and facing direction (active only when control is enabled)

### Output:
- **Motion File**: Downloadable .smpl format containing the generated motion sequence
- **Rendered Video**: MP4 visualization of the generated motion
- **Generation Logs**: Technical information about the generation process


"""
    
    def __init__(self, motion_generator: MotionGenerator):
        self.motion_generator = motion_generator
        
    def create_interface(self) -> gr.Interface:
        """Create and configure the Gradio interface."""
        return gr.Interface(
            fn=self.motion_generator.generate,
            inputs=self._create_inputs(),
            outputs=self._create_outputs(),
            title="Perpetual Motion Generation",
            description=self.ARTICLE,
        )
    
    def _create_inputs(self) -> list:
        """Create input components for the interface."""
        return [
            gr.File(label="Input File (AMASS .npz or .smpl)"),
            gr.Checkbox(label="Snap to ground", value=True),
            gr.Checkbox(label="Add random external perturbation"),
            gr.Checkbox(label="Reproject keypoints to SMPLX"),
            gr.Checkbox(label="Switch on inertialization", value=True),
            gr.Checkbox(label="Switch on control"),
            gr.Dropdown(label="Action (via induced impulses)", choices=Constants.ACTION_LIST, value='none'),
            gr.Slider(
                minimum=0, maximum=360, step=0.1, value=90,
                label="Moving orientation (degrees). 0=right, 90=outward, 180=left"
            ),
            gr.Slider(
                minimum=0, maximum=10.0, step=0.01, value=0.5,
                label="Moving speed (m/s)"
            ),
            gr.Slider(
                minimum=15, maximum=3000, step=1, value=60,
                label="Number of frames to generate (30fps)"
            ),
            gr.Slider(
                minimum=1, maximum=50, step=1, value=30,
                label="Number of denoising steps"
            ),
            gr.Slider(
                minimum=1, maximum=200, step=0.1, value=50,
                label="Guidance weight of moving velocity (only with control enabled)"
            ),
            gr.Slider(
                minimum=1, maximum=200, step=0.1, value=25,
                label="Guidance weight of facing direction (only with control enabled)"
            ),
        ]
    
    def _create_outputs(self) -> list:
        """Create output components for the interface."""
        return [
            gr.File(label="Download motion file (.npz)"),
            gr.Video(label="Rendered video"),
            gr.Textbox(label="Generation logs")
        ]




if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Autoregressive Diffusion Spatial Motion Generation Gradio Interface

This module provides a Gradio web interface for generating human motion sequences
using autoregressive diffusion models conditioned on spatial targets.
"""

import os
import sys
import glob
import random
from os.path import join as opj
from typing import Optional, Tuple, Dict

import torch
import numpy as np
import gradio as gr
from omegaconf import OmegaConf

from primal.models.motion_diffuser import ARDiffusionSpatial
from primal.utils.data_io import load_data, save_data_pkl, save_data_smplcodec


class Config:
    """Configuration constants for the application."""
    
    DEFAULT_OUTPUT_DIR = 'outputs/gradio_ARDiffusionSpatial'
    CHECKPOINT_PATTERN = 'tensorboard/version_0/checkpoints/*.ckpt'
    DATA_CONFIG_PATH = 'primal/configs/data/amass_smplx.yaml'
    DEFAULT_SUBSETS = ['SFU']
    
    # Create the output directory
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
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
        
    def load_model(self) -> ARDiffusionSpatial:
        """Load and initialize the motion diffusion model."""
        ckpt_paths = self._find_checkpoint_paths()
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints found in {self.model_dir}")
            
        latest_ckpt = ckpt_paths[-1]
        print(f'Loading model from {latest_ckpt}')
        
        self.model = ARDiffusionSpatial.load_from_checkpoint(
            latest_ckpt,
            map_location=self.device,
            strict=True
        )
        
        self.model.load_ema_parameters()
        self.model.eval()
        self.model.freeze()
        
        return self.model
        
    def _find_checkpoint_paths(self) -> list:
        """Find all checkpoint files in the model directory."""
        pattern = opj(self.model_dir, Config.CHECKPOINT_PATTERN)
        return sorted(glob.glob(pattern), key=os.path.getmtime)


class DataManager:
    """Manages data loading and configuration."""
    
    def __init__(self):
        self.config = self._load_data_config()
        
    def _load_data_config(self) -> OmegaConf:
        """Load data configuration."""
        config = OmegaConf.load(Config.DATA_CONFIG_PATH)
        config.subsets = Config.DEFAULT_SUBSETS
        config.batch_size = 1
        config.seq_len = 2  # Load entire sequence one by one
        return config
        
    def get_random_file(self) -> str:
        """Get a random file from the dataset."""
        subset = random.choice(self.config.subsets)
        seq_files = glob.glob(os.path.join(self.config.path, subset, '*/*.npz'))
        return random.choice(seq_files)


class MotionGenerator:
    """Handles motion generation logic."""
    
    def __init__(self, model: ARDiffusionSpatial, output_dir: str):
        self.model = model
        self.output_dir = output_dir
        self.data_manager = DataManager()
        os.makedirs(output_dir, exist_ok=True)
    
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
        pkl_path = save_data_pkl(self.output_dir, idx, betas, xb_gen, kpts)
        
        # Render video
        cmd = f"python primal/rendering/render_gradio.py {pkl_path} false"
        mp4_path = pkl_path + '_dist_-1.00.mp4.mp4'
        os.system(cmd)
        
        # Save SMPL file
        smpl_path = save_data_smplcodec(self.output_dir, idx, betas, xb_gen, self.model.fps)
        
        return smpl_path, mp4_path
    
    def generate(
        self,
        file: Optional[str],
        snap_to_ground: bool,
        use_vel_perturbation: bool,
        reproj_kpts: bool,
        switch_on_inertialization: bool,
        switch_on_control: bool,
        ori_angle_deg: float,
        speed: float,
        number_of_frames: int,
        number_of_inference_steps: int,
        guidance_weight_facing: float,
        guidance_weight_action: float,
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
            guidance_weight_facing=guidance_weight_facing,
            guidance_weight_action=guidance_weight_action,
        )
        
        xb_gen = xb_gen[0]
        kpts = kpts[0]
        
        # Save results
        smpl_path, mp4_path = self._save_results(betas, xb_gen, kpts)
        
        # Add target info to logs
        output_logs = output_logs + f'\nTarget: {ori_angle_deg}°'
        
        return smpl_path, mp4_path, output_logs


class GradioInterface:
    """Manages the Gradio web interface."""
    
    ARTICLE = """
## Tips for using this generator:

1. **Input File**: Upload an AMASS .npz file or a .smpl file. Only the first ≤2 frames are used.
   When no input file is uploaded, a pose with velocities is drawn from AMASS.
2. **Snap to Ground**: Enable this option to make the generated motion snap to the ground plane.
3. **Add Random External Perturbation**: Enable this option to add random external perturbation to the generated motion.
4. **Reproject keypoints to SMPLX**: Enable this option to reproject predicted joint/marker locations to SMPLX, which is more stable.
5. **Switch on Control**: Enable this option to switch on the control.
6. **Moving Orientation**: Specify the moving orientation in degrees. 0° = right, 90° = outward, 180° = left.
7. **Guidance Weight**: The larger the value, the stronger the guidance impact. Only works when 'switch on control' is enabled.

For best results, try adjusting parameters incrementally and observe the changes in the output.
"""
    
    def __init__(self, motion_generator: MotionGenerator):
        self.motion_generator = motion_generator
        
    def create_interface(self) -> gr.Interface:
        """Create and configure the Gradio interface."""
        return gr.Interface(
            fn=self.motion_generator.generate,
            inputs=self._create_inputs(),
            outputs=self._create_outputs(),
            title="Perpetual Spatial Motion Generation with Control",
            description="Generate plausible future motions based on the first frame and spatial targets",
            article=self.ARTICLE,
        )
    
    def _create_inputs(self) -> list:
        """Create input components for the interface."""
        return [
            gr.File(label="Input File (AMASS .npz or .smpl)"),
            gr.Checkbox(label="Snap to ground"),
            gr.Checkbox(label="Add random external perturbation"),
            gr.Checkbox(label="Reproject keypoints to SMPLX"),
            gr.Checkbox(label="Switch on inertialization"),
            gr.Checkbox(label="Switch on control"),
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
                minimum=1, maximum=200, step=0.1, value=0,
                label="Guidance weight of facing direction (only with control enabled)"
            ),
            gr.Slider(
                minimum=0, maximum=10, step=0.1, value=1.0,
                label="Guidance weight of the target (Scale for CFG)"
            ),
        ]
    
    def _create_outputs(self) -> list:
        """Create output components for the interface."""
        return [
            gr.File(label="Download .smpl file"),
            gr.Video(label="Rendered video"),
            gr.Textbox(label="Generation logs")
        ]


def main():
    """Main application entry point."""
    if len(sys.argv) != 2:
        print("Usage: python ARDiffusionSpatial_gradio.py <model_directory>")
        sys.exit(1)
        
    model_dir = sys.argv[1]
    device = Config.get_device()
    
    # Initialize components
    model_manager = ModelManager(model_dir, device)
    model = model_manager.load_model()
    
    motion_generator = MotionGenerator(model, Config.DEFAULT_OUTPUT_DIR)
    gradio_interface = GradioInterface(motion_generator)
    
    # Launch interface
    interface = gradio_interface.create_interface()
    interface.launch(share=True, debug=False)
    # interface.launch(server_name="127.0.0.1", server_port=8000, share=False)


if __name__ == "__main__":
    main()
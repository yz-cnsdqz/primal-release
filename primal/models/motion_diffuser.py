
from lightning import LightningModule
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


from primal.utils.mop_repr import SMPLXParser, SMPLXParserRotcont, RotConverter
from primal.models.base_models import *
from primal.models.ddpm_scheduler import DDPMScheduler




class MotionDiffuserBase(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self._setup_ema(cfg)
        self._setup_tokenizers(cfg)
        self._setup_denoiser(cfg)
        self._setup_noisescheduler(cfg)
        self._setup_additional(cfg)
        self._setup_controller(cfg)

    def _setup_controller(self, cfg):
        pass
    
    
    def _setup_ema(self, cfg):
        """Setup exponential moving average of model weights.

        EMA is applied at every epoch if enabled. This helps stabilize training
        and often produces better results during inference.

        Args:
            cfg: Configuration object containing EMA parameters.
                - ema_decay (float): Decay rate for EMA updates. Default: 0.999
                - ema_start_epoch_ratio (float): Fraction of total epochs before starting EMA. Default: 0.0
        """
        self.ema_decay = cfg.get('ema_decay', 0.999)
        self.ema_start_epoch_ratio = cfg.get('ema_start_epoch_ratio', 0.0)
        self.ema_weights = None
    
    def initialize_ema_weights(self):
        """Initialize EMA weights with the model's current parameters.

        Creates a copy of model parameters for exponential moving average tracking,
        excluding CLIP model parameters which are typically frozen.

        Note:
            This should be called before starting training. Parameters starting with
            'cliptextmodel' or 'cliptokenizer' are excluded from EMA tracking.
        """
        self.ema_weights = OrderedDict(
            (name, param.clone().detach())
            for name, param in self.named_parameters()
            if not (name.startswith('cliptextmodel') or name.startswith('cliptokenizer'))
        )

    def update_ema_weights(self):
        """Update EMA weights using the current model parameters.

        Applies exponential moving average update: ema = decay * ema + (1 - decay) * current

        Raises:
            ValueError: If EMA weights have not been initialized via initialize_ema_weights().

        Note:
            This should be called at the end of each epoch during training.
        """
        if self.ema_weights is None:
            raise ValueError("EMA weights have not been initialized. Call initialize_ema_weights() before training.")

        for name, model_param in self.named_parameters():
            if name in self.ema_weights:
                self.ema_weights[name].data.mul_(self.ema_decay).add_((1 - self.ema_decay) * model_param.data)


    def apply_ema_weights(self):
        """Replace model parameters with EMA weights.

        Copies EMA-tracked weights into the model's actual parameters, typically used
        for inference or validation to benefit from the stabilizing effect of EMA.

        Raises:
            ValueError: If EMA weights are not initialized.
        """
        if self.ema_weights is None:
            raise ValueError("EMA weights are not initialized.")

        for name, param in self.named_parameters():
            if name in self.ema_weights:
                param.data.copy_(self.ema_weights[name].data)



    def _setup_tokenizers(self, cfg):
        """Setup SMPL-X tokenizers for motion representation parsing.

        Initializes the appropriate SMPL-X parser based on the motion representation type.
        Supports both standard axis-angle and rotation continuity representations.

        Args:
            cfg: Configuration object containing:
                - motion_repr (str): Motion representation type (e.g., 'smplx_jts_locs_velocity_rotcont')
                - data.model_reg_path (str): Path to SMPL-X model registry (fallback)

        Note:
            Uses MODEL_REGISTRY_PATH environment variable if set, otherwise falls back to cfg.data.model_reg_path.
            All parsers use SSM2.json for marker definitions and 16 beta shape parameters.

        Raises:
            ValueError: If MODEL_REGISTRY_PATH is not set and model_reg_path is not in config.
        """
        mrepr = cfg.get('motion_repr', 'smplx_jts_locs_vel')

        # Use MODEL_REGISTRY_PATH environment variable if available, otherwise fall back to config
        model_reg_path = os.getenv('MODEL_REGISTRY_PATH', getattr(cfg.data, 'model_reg_path', None))
        if model_reg_path is None:
            raise ValueError("MODEL_REGISTRY_PATH environment variable not set and model_reg_path not found in config")

        # body model
        if 'rotcont' in mrepr:
            self.smplx_parser = SMPLXParserRotcont(osp.join(model_reg_path,"models/SMPLX/neutral/SMPLX_neutral.npz"),
                                        "primal/utils/SSM2.json",
                                        num_betas=16)
        else:
            self.smplx_parser = SMPLXParser(osp.join(model_reg_path,"models/SMPLX/neutral/SMPLX_neutral.npz"),
                                            "primal/utils/SSM2.json",
                                            num_betas=16)
        self.smplx_parser.eval()
        
        

    def _setup_denoiser(self, cfg):
        """Setup the denoising network architecture.

        This method should be implemented by subclasses to define the specific
        denoiser architecture (e.g., Transformer-based models).

        Args:
            cfg: Configuration object containing network architecture parameters.
        """
        pass

    def _setup_additional(self, cfg):
        """Setup additional model components and hyperparameters.

        This method can be overridden by subclasses to initialize task-specific
        components like loss functions, data canonicalization settings, etc.

        Args:
            cfg: Configuration object containing additional model settings.
        """
        pass

    def _setup_noisescheduler(self, cfg):
        """Setup the noise scheduler for the diffusion process.

        Initializes the noise scheduler that controls the forward and reverse
        diffusion processes during training and inference.

        Args:
            cfg: Configuration object containing scheduler parameters:
                - scheduler.type (str): Scheduler type (currently only 'ddpm' supported)
                - scheduler.num_train_timesteps (int): Number of diffusion steps
                - scheduler.beta_schedule (str): Beta schedule type (e.g., 'linear', 'scaled_linear')
                - scheduler.prediction_type (str): What the model predicts ('epsilon', 'sample', etc.)

        Raises:
            NotImplementedError: If scheduler type is not 'ddpm'.
        """
        if cfg.scheduler.type == 'ddpm':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.scheduler.num_train_timesteps,
                beta_schedule=cfg.scheduler.beta_schedule,
                prediction_type=cfg.scheduler.prediction_type,
                clip_sample=False,
                )
        else:
            raise NotImplementedError




    def configure_optimizers(self):
        """Configure optimizers for training.

        Sets up the AdamW optimizer for model parameter updates. This method is called
        by PyTorch Lightning during training setup.

        Returns:
            dict: Dictionary containing the optimizer with key "optimizer".
        """
        # can return multiple optimizers and learning_rate schedulersq
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return {"optimizer":optimizer,}



    def training_step(self,batch):
        """Execute a single training step.

        Performs forward pass, computes losses, logs metrics, and updates EMA weights
        if enabled and past the start epoch threshold.

        Args:
            batch (dict): Batch of training data containing motion sequences and metadata.

        Returns:
            dict: Dictionary of computed losses including 'loss' (total) and component losses.

        Note:
            - Logs all losses with '/train' suffix for TensorBoard
            - Initializes EMA weights on first eligible epoch
            - Updates EMA weights on subsequent epochs after initialization
        """
        losses = self.forward(batch)
        for k,v in losses.items():
            self.log(
                f"{k}/train", v,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )

        ## apply EMA
        ema_start_epoch = self.trainer.max_epochs*self.ema_start_epoch_ratio
        if self.ema_decay > 0 and self.current_epoch>ema_start_epoch:
            if self.ema_weights is None:
                self.initialize_ema_weights()
            else:
                self.update_ema_weights()

        return losses

    

    @torch.no_grad()
    def validation_step(self, batch):
        """Execute a single validation step.

        Evaluates the model on validation data using both standard parameters and
        EMA parameters (if available), logging both sets of metrics.

        Args:
            batch (dict): Batch of validation data containing motion sequences and metadata.

        Returns:
            dict: Dictionary containing:
                - normal_losses (dict): Losses computed with standard model parameters
                - ema_losses (dict or None): Losses computed with EMA parameters if available

        Note:
            - Logs standard losses with '/val' suffix
            - Logs EMA losses with '/val_ema' suffix
            - Temporarily swaps in EMA weights for evaluation, then restores original weights
        """
        # Evaluate with normal model parameters
        normal_losses = self.forward(batch)
        for k, v in normal_losses.items():
            self.log(f"{k}/val", v, on_step=False, on_epoch=True, sync_dist=True)

        # Temporarily replace with EMA parameters
        if self.ema_weights is not None:
            original_weights = OrderedDict((name, param.clone()) for name, param in self.named_parameters())
            self.apply_ema_weights()
            try:
                # Evaluate with EMA parameters
                ema_losses = self.forward(batch)
                for k, v in ema_losses.items():
                    self.log(f"{k}/val_ema", v, on_step=False, on_epoch=True, sync_dist=True)
            finally:
                # Restore original parameters
                for name, param in self.named_parameters():
                    param.data.copy_(original_weights[name].data)

        return {"normal_losses": normal_losses, "ema_losses": ema_losses if self.ema_weights else None}



    def forward(self, batch):
        pass



    def load_ema_parameters(self, ):
        """
        Load EMA parameters into the current model.

        Args:
            ema_state_dict (OrderedDict): EMA-weighted parameters.
        """
        if self.ema_weights is not None:
            for name, param in self.named_parameters():
                if name in self.ema_weights:
                    param.data.copy_(self.ema_weights[name].data)
            print("--EMA weights successfully applied.")
        else:
            print("--no EMA weights. Nothing to apply.")

            

    def on_save_checkpoint(self, checkpoint):
        """
        Modify the checkpoint to include both EMA and non-EMA parameters.

        Args:
            checkpoint (dict): The checkpoint dictionary to save.
        """
        # Remove CLIP model parameters from the checkpoint
        if 'state_dict' in checkpoint:
            keys_to_remove = [
                key for key in checkpoint['state_dict'].keys()
                if key.startswith('cliptextmodel') or key.startswith('cliptokenizer')
            ]
            for key in keys_to_remove:
                checkpoint['state_dict'].pop(key)

        # Add EMA parameters to the checkpoint
        if self.ema_weights is not None:
            checkpoint['ema_state_dict'] = {
                name: param.clone().detach()
                for name, param in self.ema_weights.items()
            }

    def on_load_checkpoint(self, checkpoint):
        """
        Modify the checkpoint loading process to handle both EMA and non-EMA parameters.

        Args:
            checkpoint (dict): The checkpoint dictionary to load.
        """
        if 'ema_state_dict' in checkpoint:
            self.ema_weights = OrderedDict(
                (name, checkpoint['ema_state_dict'][name].clone().to(self.device))
                for name in checkpoint['ema_state_dict'].keys()
            )
            
       








""" 
#################################################################################
autoregressive diffusion models 
#################################################################################
"""


from primal.utils.mop_repr import inertialize

class ARDiffusion(MotionDiffuserBase):
    def _setup_denoiser(self, cfg):
        self.mrepr = cfg.get('motion_repr', 'smplx_jts_locs_vel')
        if self.mrepr=='smplx_jts_locs_velocity':
            # root + pose + joint locations
            self.x_dim = x_dim = 3+3 + 21*3 + 22*6
            self.n_dim_rot=3
            self.n_kpts = 22
        elif self.mrepr=='smplx_jts_locs_velocity_rotcont':
            # root + pose + joint locations
            self.x_dim = x_dim = 3+6 + 21*6 + 22*6
            self.n_dim_rot=6
            self.n_kpts = 22
        elif self.mrepr=='smplx_ssm67_locs_velocity':
            # root + pose + ssm2 marker locations
            self.x_dim = x_dim = 3+3 + 21*3 + 67*6
            self.n_dim_rot=3
            self.n_kpts = 67
        else:
            raise NotImplementedError('wrong mrepr')
        
        ## transformer
        self.modeltype = cfg.network.get('type', 'transformerAdaLN0')
        if self.modeltype=='transformerAdaLN0':
            self.denoiser = TransformerAdaLN0(
                x_dim,
                x_dim,
                cfg.network.h_dim,
                cfg.network.n_layer,
                cfg.network.n_head,
                dropout=cfg.network.dropout,
                n_time_embeddings = cfg.scheduler.num_train_timesteps,
                use_positional_encoding=cfg.network.get('use_positional_encoding', True),
            )
        elif self.modeltype=='transformerInContext':
            self.denoiser = TransformerInContext(
                x_dim,
                x_dim,
                cfg.network.h_dim,
                cfg.network.n_layer,
                cfg.network.n_head,
                dropout=cfg.network.dropout,
                n_time_embeddings = cfg.scheduler.num_train_timesteps,
                separate_condition_token=cfg.network.get('separate_condition_token', True),
                use_positional_encoding=cfg.network.get('use_positional_encoding', True),
                act_fun=cfg.network.get('act_fun', 'relu'),
            )
        else:
            raise NotImplementedError


        
    def aa2rotcont(self, xb):
        nb, nt = xb.shape[:2]
        # process smplx params
        transl, glorot_aa = xb[:,:,:3], xb[:,:,3:6]
        global_rotcont = RotConverter.aa2cont(glorot_aa)
        pose_aa = xb[:,:,6:]
        pose_rotcont = RotConverter.aa2cont(pose_aa.reshape(nb,nt,-1,3)).reshape(nb,nt,-1)
        
        # cat
        mpattern = torch.cat([transl, global_rotcont, pose_rotcont],dim=-1)
        
        return mpattern


    def rotcont2aa(self, xb):
        nb, nt = xb.shape[:2]
        # process smplx params
        transl = xb[...,:3]
        glorot_cont = xb[...,3:9]
        pose_rotcont = xb[...,9:9+21*6]
        glorot_aa = RotConverter.cont2aa(glorot_cont)
        pose_aa = RotConverter.cont2aa(pose_rotcont.reshape(nb,nt,-1,6)).reshape(nb,nt,-1)
        xb_new = torch.cat([transl, glorot_aa, pose_aa],dim=-1)

        return xb_new

    
    def _setup_additional(self, cfg):
        ## re data canonicalization
        if cfg.get('canonical_tidx', 'start')=='start':
            self.canonical_tidx = 0
        elif cfg.get('canonical_tidx', 'start')=='middle':
            self.canonical_tidx = cfg.data.seq_len//2
        elif cfg.get('canonical_tidx', 'start')=='end':
            self.canonical_tidx = -1
        else:
            raise NotImplementedError
        
        ## re the loss definition
        self.use_l1_norm_fk = cfg.get('use_l1_norm_fk', False)
        self.use_l1_norm_vel = cfg.get('use_l1_norm_vel', False)

        ## data framerate
        self.fps = self.hparams.data.framerate
        self.use_metric_velocity = self.hparams.get('use_metric_velocity', False)

        
        ## noised scheduled residual
        self.ts = torch.linspace(
            0,1, 
            self.hparams.data.seq_len-1, # -1 because we compute the velocity
            device=self.device
        )
        
    def _setup_controller(self, cfg):
        ## condition embedding
        self.emb_motionseed = nn.Linear(self.x_dim, cfg.network.h_dim)

    def _fwd_smplx_seq(self, betas, xb, return_ssm2=False):
        """perform smplx fk with batch sequence data

        Args:
            betas (_type_): [b,t,d]
            xb (_type_): [b,t,d]

        Returns:
            jts_body: [b,t,J,d], J=22
            markers: [b,t,p,d], p=67
        """
        nb, nt = betas.shape[:2]
        betas_reshape = betas.reshape(nb*nt,-1)
        xb_reshape = xb.reshape(nb*nt,-1)

        jts, markers = self.smplx_parser.forward_smplx(
            betas_reshape, 
            xb_reshape[:,:3], 
            xb_reshape[:,3:3+self.n_dim_rot],
            xb_reshape[:,3+self.n_dim_rot:],
            returntype='jts_and_ssm2_markers'
            )
        jts = jts.reshape(nb,nt,-1,3)[:,:,:22]
        markers = markers.reshape(nb,nt,-1,3)
        
        if 'ssm' in self.mrepr or return_ssm2:
            return markers
        elif 'jts' in self.mrepr:
            return jts
        else:
            raise NotImplementedError('wrong mrepr')


    def canonicalization(
            self, betas, xb, 
            return_transf=False,
            place_on_ground=False,
        ):
        """1) motion canonicalization based on the body coordinate in the first frame
           2) motion encoding based on the body parameters and the joints locations
        """
        canonical_tidx = 0
        rotmat_c, transl_c = self.smplx_parser.get_new_coordinate(
            betas[:,canonical_tidx], 
            xb[:,canonical_tidx],
            coordinate_on_ground=place_on_ground,
        )
        
        ## obtain updated smplx paramters
        xb_c = self.smplx_parser.update_transl_glorot_seq(rotmat_c, transl_c, betas, xb)
        
        if return_transf:
            return xb_c, rotmat_c, transl_c
        else:
            return xb_c




    def snap_init_cond_to_ground(self, betas, xb):
        jts_w = self.smplx_parser.forward_smplx(
            betas[:,0], 
            xb[:,0,:3], 
            xb[:,0,3:3+self.n_dim_rot],
            xb[:,0,3+self.n_dim_rot:],
            returntype='jts'
        )
        
        heights = jts_w[...,1]
        heights_flatten = heights.view(heights.size(0), -1)
        ## compute body ground contact
        if torch.all(heights_flatten>0):
            tpidx_flatten = heights_flatten.abs().argmin(dim=1)
        else:    
            tpidx_flatten = heights_flatten.argmin(dim=1)
        
        dist2ground = heights_flatten[:,tpidx_flatten].item()
        ## projection
        xb[...,1] -= (dist2ground - 0.01)    
        
        return xb




    def forward_one(self,batch):
        
        (
            betas_all, 
            xb_all, 
         ) = (
                batch["betas"], 
                batch["xb"], 
         )
        
        losses = {}
        fn_dist_fk = F.l1_loss if self.use_l1_norm_fk else F.mse_loss
        fn_dist_vel = F.l1_loss if self.use_l1_norm_vel else F.mse_loss
        
        # data preprcessing and compute velocity
        with torch.no_grad():
            betas = betas_all
            ## process smplx params
            if 'rotcont' in self.mrepr:
                xb_all = self.aa2rotcont(xb_all)
            
            ## motion encoding
            xb = self.canonicalization(
                betas, xb_all,
                return_transf=False
            )
            kpts = self._fwd_smplx_seq(betas, xb)
            vel = kpts[:,1:] - kpts[:,:-1]
            if self.use_metric_velocity:
                vel = vel * self.fps
            xb = xb[:,:-1]
            kpts = kpts[:,:-1]
            nb, nt = xb.shape[:2]
            
            xs = torch.cat(
                [
                    xb, 
                    kpts.reshape(nb, nt, -1),
                    vel.reshape(nb, nt, -1)
                ],dim=-1)
            
            ## add noise to the motion seed
            xs_seed = xs[:,:1].detach().clone()
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (nb,), 
                device=self.device,
                dtype=torch.int64
            )
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            xs_noise = self.noise_scheduler.add_noise(xs, torch.randn_like(xs), timesteps)
            
        ## fwd pass of motion model
        cond_emb = self.emb_motionseed(xs_seed)
        xs_pred = self.denoiser(
            timesteps, 
            xs_noise,
            c_emb=cond_emb,
            attn_mask = None,
        )
        
        
        ## decompose prediction
        xb_pred = xs_pred[...,:-self.n_kpts*6]
        
        
        # Compute the loss
        ## denoiser loss
        if self.hparams.scheduler.get('prediction_type', 'epsilon') == 'sample':
            losses['loss_simple'] = F.mse_loss(xs_pred, xs)
        else:
            raise NotImplementedError

        ## forward kinematics
        kpts_pred_fk = self._fwd_smplx_seq(betas[:,:nt], xb_pred)
        losses['loss_fk'] = fn_dist_fk(kpts_pred_fk, kpts)
        ## velocity consistency
        vel_pred_fk = kpts_pred_fk[:,1:] - kpts_pred_fk[:,:-1]
        if self.use_metric_velocity:
            vel_pred_fk = vel_pred_fk * self.fps
        losses['loss_vel'] = fn_dist_vel(vel[:,:-1], vel_pred_fk)
                
        ## total loss
        losses['loss']= self.hparams.weight_simple*losses['loss_simple']  \
            + self.hparams.weight_fk*losses['loss_fk'] \
            + self.hparams.weight_vel*losses['loss_vel']
        
        return losses



    def forward(self,batch):
        if self.hparams.use_ss:
            raise NotImplementedError('scheduled sampling not implemented yet')
        else:
            losses = self.forward_one(batch)
        
        return losses


    def on_train_epoch_end(self):
        ## enable scheduled sampling later
        ss=(self.hparams.ss_end_epoch-self.current_epoch)/(self.hparams.ss_end_epoch-self.hparams.ss_start_epoch)
        self.ss_ratio = min(max(ss,0),1)




    @torch.no_grad()
    def generate_perpetual_navigation(
            self,
            batch,
            n_inference_steps=10,
            nt_max = 1200,
            guidance_weight_mv=50,
            guidance_weight_facing=25,
            reproj_kpts=False,
            snap_to_ground=False,
            use_vel_perburbation=False,
            switch_on_control=False,
            switch_on_inertialization=True,
            perform_principled_action=None,
        ):
        """Generate perpetual motion autoregressively with optional navigation control.

        This method generates long-horizon motion sequences by autoregressively applying
        the diffusion model in a sliding window fashion. Each iteration generates a motion
        primitive that is seamlessly connected to the previous one through canonicalization
        and optional inertialization.

        The generation process involves four main stages per iteration:
        1. Canonicalization: Transform motion to local coordinate frame
        2. Control Embedding: Encode motion seed and optional control signals
        3. Reverse Diffusion: Denoise random noise to generate motion primitive
        4. Post-processing: Transform back to world coordinates and apply constraints

        Args:
            batch (dict): Input batch containing:
                - betas (torch.Tensor): SMPL-X shape parameters [1, T, 16]
                - xb (torch.Tensor): Initial SMPL-X pose parameters [1, T, D]
                - ori (torch.Tensor): Desired orientation/direction trajectory [1, T, 3]

            n_inference_steps (int, optional): Number of diffusion denoising steps per primitive.
                Lower values are faster but may reduce quality. Defaults to 10.

            nt_max (int, optional): Maximum number of frames to generate in total.
                Generation stops when this limit is reached. Defaults to 1200.

            guidance_weight_mv (float, optional): Classifier guidance weight for average
                movement velocity direction. Higher values enforce stronger adherence to
                desired trajectory direction. Only used if switch_on_control=True. Defaults to 50.

            guidance_weight_facing (float, optional): Classifier guidance weight for body
                facing direction. Controls how strongly the body faces the movement direction.
                Only used if switch_on_control=True. Defaults to 25.

            reproj_kpts (bool, optional): If True, recompute keypoints via forward kinematics
                instead of transforming cached keypoints. More accurate but slower. Defaults to False.

            snap_to_ground (bool, optional): If True, projects motion to ensure contact with
                ground plane (y=0.01). Prevents floating artifacts. Defaults to False.

            use_vel_perburbation (bool, optional): If True, applies velocity perturbations
                to seed motion based on perform_principled_action. Used for action generation.
                Defaults to False.

            switch_on_control (bool, optional): If True, enables classifier-free guidance
                for trajectory following (movement and facing direction). Defaults to False.

            switch_on_inertialization (bool, optional): If True, applies inertialization
                (smooth blending) between consecutive motion primitives to reduce artifacts.
                Highly recommended for smooth transitions. Defaults to True.

            perform_principled_action (str or None, optional): Specifies a principled action
                to perform by applying targeted velocity perturbations. Options:
                - 'none': No perturbation
                - 'left_kick': Kick with left leg
                - 'right_kick': Kick with right leg
                - 'run_forward': Run forward motion
                - 'flip_back': Backflip motion
                - 'roll_forward': Forward roll motion
                If not None and not 'none', automatically sets use_vel_perburbation=True.
                Defaults to None.

        Returns:
            tuple: A 4-tuple containing:
                - betas (torch.Tensor): SMPL-X shape parameters [1, 1, 16]
                - xb_gen_list (list): List with generated SMPL-X pose sequences [1, T_gen, D]
                - kpts_vis_list (list): List with keypoint+velocity visualizations [1, T_gen, N, 4]
                - outputlogs (str): Runtime statistics and performance metrics

        Note:
            - The method uses autoregressive generation with overlapping windows
            - Runtime is logged for each stage: canonicalization, control_embedding,
              reverse_diffusion, and post_processing
            - Velocity rays are generated for visualization purposes
            - The last frame of each primitive is used to seed the next iteration
        """

        # setup data io
        (
            betas_all, 
            xb_all,
            ori,
         ) = (
                batch["betas"], 
                batch["xb"], 
                batch["ori"],
         )
        
        # setup denoiser
        self.noise_scheduler.set_timesteps(n_inference_steps)
        nt_tw = self.hparams.data.seq_len-1
        if not self.use_metric_velocity:
            ori = ori / self.fps


        if perform_principled_action is not None:
            if perform_principled_action == 'none':
                use_vel_perburbation = False
            else:
                use_vel_perburbation = True
        
        ###################  loop of recursion #################
        tt = 0
        xb_gen = []
        kpts_gen = []
        vel_gen = []
        vel_gen_avg = []

        nb = 1    
        betas = betas_all[:,:1]

        runtimelog = {
            'canonicalization': [],
            'control_embedding': [],
            'reverse_diffusion': [],
            'post_processing': [],
        }
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        while tt < nt_max:
            if tt + nt_tw > nt_max:
                break

            ################### 1. canonicalization ################
            start_event.record()
            if tt==0:
                if 'rotcont' in self.mrepr:
                    xb_all = self.aa2rotcont(xb_all)

                if snap_to_ground:
                    xb_all = self.snap_init_cond_to_ground(betas_all, xb_all)

                xb, rotmat_c, transl_c = self.canonicalization(
                    betas_all, xb_all,
                    return_transf=True
                )
                kpts = self._fwd_smplx_seq(betas_all, xb)
                vel_seed_c = kpts[:,1:] - kpts[:,:-1]
                if self.use_metric_velocity:
                    vel_seed_c = vel_seed_c * self.fps

                xb_seed_c = xb[:,:-1]
                kpts_seed_c = kpts[:,:-1]
                
            else:
                ## recompute keypoint locations and velocities
                xb_seed_w = xb_gen_w_[:,-1:].detach().clone()
                xb_seed_c, rotmat_c, transl_c = self.canonicalization(
                    betas.repeat(1,xb_seed_w.shape[1],1), xb_seed_w,return_transf=True)
                
                if reproj_kpts:
                    kpts_seed_c = self._fwd_smplx_seq(
                        betas.repeat(1,xb_seed_c.shape[1],1), xb_seed_c
                    )
                else:
                    kpts_seed_w = kpts_gen_w_[:,-1:].detach().clone()
                    kpts_seed_c = torch.einsum(
                        'bij,btpj->btpi', 
                        rotmat_c.permute(0,2,1), 
                        kpts_seed_w-transl_c.unsqueeze(-2)
                    )
                
                vel_seed_w = vel_gen_w_[:,-1:].detach().clone()
                vel_seed_c = torch.einsum(
                    'bij,btpj->btpi', 
                    rotmat_c.permute(0,2,1),
                    vel_seed_w
                )

            if use_vel_perburbation and tt==0:
                if perform_principled_action=='left_kick':
                    idx = [4,7] # left leg kick
                    perturb = torch.tensor([[[0,0,1]]]).to(self.device) * 0.5
                    vel_seed_c[:,:,idx] += perturb
                elif perform_principled_action=='right_kick':
                    idx = [5,8] # right leg kick
                    perturb = torch.tensor([[[0,0,1]]]).to(self.device) * 0.5
                    vel_seed_c[:,:,idx] += perturb
                elif perform_principled_action=='run_forward':
                    idx = [0,12,16,17] # pelvis, neck, shoulders
                    perturb = torch.tensor([[[0,0,1]]]).to(self.device) * 0.5
                    vel_seed_c[:,:,idx] += perturb
                elif perform_principled_action=='flip_back':
                    idx = [15] # head
                    perturb = torch.tensor([[[0,0,-1]]]).to(self.device) * 1
                    vel_seed_c[:,:,idx] += perturb
                elif perform_principled_action=='roll_forward':
                    idx = [15, 16, 17, 18, 19] # head and shoulders and elbows
                    perturb = torch.tensor([[[0,-1,1]]]).to(self.device) * 0.5
                    vel_seed_c[:,:,idx] += perturb

            end_event.record()
            torch.cuda.synchronize()
            runtimelog['canonicalization'].append(start_event.elapsed_time(end_event))


            ################### 2. control embedding ################
            start_event.record()
            
            ## obtain guidance velocity
            ori_c = torch.einsum('bij,btj->bti', rotmat_c.permute(0,2,1),ori)
            
            xs_seed = torch.cat([xb_seed_c, 
                                kpts_seed_c.reshape(nb, 1, -1),
                                vel_seed_c.reshape(nb, 1, -1)],
                                dim=-1)
            
            ## encode of the motion seed
            cond_emb = self.emb_motionseed(xs_seed)
        
            end_event.record()
            torch.cuda.synchronize()
            runtimelog['control_embedding'].append(start_event.elapsed_time(end_event))
            
            ################### 3. reverse diffusion ################
            start_event.record()
            input = torch.randn(nb, nt_tw, self.x_dim).to(self.device) # batch cond and uncond

            for i, t in enumerate(self.noise_scheduler.timesteps):
                ## set time steps and the first frame
                t_tensor = t.to(self.device).unsqueeze(0).repeat(input.shape[0])
                
                ## denoising without condition
                modeloutput = self.denoiser(
                    t_tensor, 
                    input,
                    c_emb=cond_emb,
                    attn_mask = None,
                    )
                
                # classifier guidance on the velocities
                # to speed up, we derive the analytical gradients
                ## 1) the body is supposed to move in the direction of ori_c, L = (x-ori_c)^2
                ## 2) the body is supposed to face in the direction of ori_c, the last pose facing ori_c.
                ##     namely, L = torch.sum( (cross(x/||x||, y) - ori_c/||ori_c|| )**2 )
                ## when using metric velocity, the guidance weight should be 0.01 times smaller
                if switch_on_control:
                    ## 1) moving direction, based on average velocity.
                    ## This guidance performs worse if t_pred is shorter, since the averaging becomes less sound.
                    vels_reshaped = modeloutput[...,-3*self.n_kpts:].reshape(nb, -1, self.n_kpts, 3)
                    ### on the mean velocity
                    grad_mv_vels = 2*(vels_reshaped.mean(dim=[1,2],keepdim=True) - ori_c.unsqueeze(-2)) #[nb,1,1,3]
                    # grad_mv_vels = grad_mv_vels / (nt_tw + self.n_kpts)
                    grad_mv_vels = grad_mv_vels.repeat(1, nt_tw, self.n_kpts,1).reshape(nb, nt_tw, -1)
                    grad_mv_pad = torch.zeros_like(input[...,:-3*self.n_kpts])
                    grad_mv = torch.cat([grad_mv_pad, grad_mv_vels], dim=-1)

                    ## 2) facing direction. The last pose in the primitive is facing ori_c.
                    ## the following implementation of grad_p is identical to autodiff, already verified
                    ## guidance weight= 25-100 can give good results. When reprojecting keypoints, this weight can be smaller.
                    ### 2.1) specify input
                    ori_cn = ori_c / (ori_c.norm(dim=-1,keepdim=True) + 1e-6)
                    # ori_cn = ori / (ori.norm(dim=-1,keepdim=True) + 1e-6)
                    kpts_reshaped = modeloutput[...,-6*self.n_kpts:-3*self.n_kpts].reshape(nb, -1, self.n_kpts, 3)
                    kpts_last = kpts_reshaped[:,-1]
                    x_axis = kpts_last[:,1] - kpts_last[:,2]
                    x_axis[...,1] = 0
                    ### 2.2) calc gradient
                    x_axis_norm = x_axis.norm(dim=-1, keepdim=True)
                    y_axis = torch.tensor([[0,1,0]]).float().to(self.device).repeat(nb, 1)
                    ssmat_y = skew_symmetric_matrix(y_axis)
                    idty = torch.eye(3).to(self.device).unsqueeze(0)
                    grad_normalization = (idty - (x_axis.unsqueeze(-1) @ x_axis.unsqueeze(-2))/(x_axis_norm**2))/x_axis_norm
                    grad_p = 2*(x_axis/x_axis_norm @ ssmat_y -ori_cn) @ (-ssmat_y) @ grad_normalization
                    ### 2.3) assign gradients to the input
                    grad_kpts_all = torch.zeros_like(kpts_reshaped)
                    grad_kpts_all[:,-1:,1] = grad_p
                    grad_kpts_all[:,-1:,2] = -grad_p
                    grad_facing = torch.zeros_like(input)
                    grad_facing[...,-6*self.n_kpts:-3*self.n_kpts] = grad_kpts_all.reshape(nb, nt_tw, -1)

                    ## combine the gradients
                    guidance_grad_ori = -(guidance_weight_facing * grad_facing 
                                            + guidance_weight_mv * grad_mv)
                else:
                    guidance_grad_ori = None
                
                
                # step
                stepoutput = self.noise_scheduler.step(
                    modeloutput, t, input, 
                    guidance=guidance_grad_ori
                )
                input = stepoutput.prev_sample
                

            ## project to above the ground floor
            if snap_to_ground: 
                xb_pred = input[:,:,:-self.n_kpts*6]
                jts_c = self._fwd_smplx_seq(betas.repeat(1, nt_tw, 1), xb_pred)
                jts_w = torch.einsum('bij,btpj->btpi', rotmat_c, jts_c)+transl_c.unsqueeze(-2)
                heights = jts_w[...,1]
                heights_flatten = heights.view(heights.size(0), -1)
                ## compute body ground contact
                if torch.all(heights_flatten>0):
                    tpidx_flatten = heights_flatten.abs().argmin(dim=1)
                else:    
                    tpidx_flatten = heights_flatten.argmin(dim=1)
                
                dist2ground = heights_flatten[:,tpidx_flatten].item()
                ## projection
                input[...,1] -= (dist2ground - 0.01)    

            # inertialization
            if switch_on_inertialization:
                output = inertialize(xs_seed, input, omega=10.0, dt=1.0/30.0)
            else:
                output = input
            
            end_event.record()
            torch.cuda.synchronize()
            runtimelog['reverse_diffusion'].append(start_event.elapsed_time(end_event))
            
            ################### 4. transform back to world coordinate ################
            start_event.record()
            output_betas = betas.repeat(1, nt_tw, 1)
            
            xb_gen_c = output[...,:-self.n_kpts*6]
            kpts_gen_c = output[...,-self.n_kpts*6:-self.n_kpts*3].reshape(nb,-1,self.n_kpts,3)
            vel_gen_c = output[...,-self.n_kpts*3:].reshape(nb,-1,self.n_kpts,3)
            
            ## transform back to the original world coordinate
            xb_gen_w_ = self.smplx_parser.update_transl_glorot_seq(
                rotmat_c, transl_c, output_betas, xb_gen_c, fwd_transf=True)
            kpts_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, kpts_gen_c) + transl_c.unsqueeze(-2)
            vel_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, vel_gen_c)
            
            xb_gen.append(xb_gen_w_[:,:-1].detach().clone())
            kpts_gen.append(kpts_gen_w_[:,:-1].detach().clone())
            vel_gen.append(vel_gen_w_[:,:-1].detach().clone())
            vel_gen_avg.append(vel_gen_w_[:,:-1].mean(dim=[1,2],keepdim=True).repeat(1,nt_tw-1,1,1).detach().clone())
            
            end_event.record()
            torch.cuda.synchronize()
            runtimelog['post_processing'].append(start_event.elapsed_time(end_event))

            tt += nt_tw-1


        # cat all sequences
        xb_gen = torch.cat(xb_gen, dim=1)
        if 'rotcont' in self.mrepr:
            xb_gen_aa = self.rotcont2aa(xb_gen)
        else:
            xb_gen_aa = xb_gen
        kpts_gen = torch.cat(kpts_gen, dim=1)
        vel_gen = torch.cat(vel_gen, dim=1)
        vel_gen_avg = torch.cat(vel_gen_avg, dim=1)
        if not self.use_metric_velocity:
            vel_gen *= self.fps
            vel_gen_avg *= self.fps
        
        ################# generate rays: start ######################
        # Normalize the velocity vectors to get the directions
        vel_dir = vel_gen / vel_gen.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Define the step size for each point in the ray
        step_size = 0.02  # You can adjust this value as needed
        steps = torch.arange(0, 10).view(1, 1, 1, 1, -1).to(self.device)  # Shape: [1, 1, 1, 10]

        # Multiply directions by the step size and steps to create rays
        rays = kpts_gen.unsqueeze(-1) + vel_dir.unsqueeze(-1) * steps * step_size  # Shape: [b, t, p, 3, 10]

        # Rearrange the rays if necessary (e.g., [t, p, 10, 3] for compatibility with other operations)
        rays = rays.permute(0, 1, 2, 4, 3) # Now shape is [b, t, p,10, 3]
        velnorms = vel_gen.norm(dim=-1,keepdim=True).unsqueeze(-1).repeat(1,1,1,10,1)
        kpts_gen_vis = torch.cat([rays, velnorms], dim=-1)
        kpts_gen_vis = kpts_gen_vis.reshape(nb,kpts_gen.shape[1],-1,4)
        ################# generate rays: end ######################

        ################# Compute and visualize averaged velocity ######################
        # vel_gen_avg = vel_gen.mean(dim=-2,keepdim=True)
        kpts_gen_avg = kpts_gen[:,:,:1].clone()
        kpts_gen_avg[...,1] = 1.7 # place pelvis to ground
        vel_dir = vel_gen_avg / vel_gen_avg.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Define the step size for each point in the ray
        step_size = 0.05  # You can adjust this value as needed
        steps = torch.arange(0, 20).view(1, 1, 1, 1, -1).to(self.device)  # Shape: [1, 1, 1, 10]
        
        # Multiply directions by the step size and steps to create rays
        rays = kpts_gen_avg.unsqueeze(-1) + vel_dir.unsqueeze(-1) * steps * step_size  # Shape: [b, t, p, 3, 10]

        # Rearrange the rays if necessary (e.g., [t, p, 10, 3] for compatibility with other operations)
        rays = rays.permute(0, 1, 2, 4, 3) # Now shape is [b, t, p,10, 3]
        velnorms = vel_gen_avg.norm(dim=-1,keepdim=True).unsqueeze(-1).repeat(1,1,1,20,1)
        kpts_gen_avg = torch.cat([rays, velnorms], dim=-1)
        kpts_gen_avg = kpts_gen_avg.reshape(nb,kpts_gen_avg.shape[1],-1,4)
        # kpts_gen_vis = torch.cat([kpts_gen_vis, kpts_gen_avg], dim=-2)


        # stats runtime
        outputlogs = f"avarage runtime to generate a motion primitive ({nt_tw} frames, {1000*nt_tw/30.:.3f} ms ): \n"
        avg = 0
        for k, v in runtimelog.items():
            outputlogs += f"-- {k}: {np.mean(v):.3f} ms\n"
            avg += np.mean(v)
        outputlogs += f"-- total: {avg:.3f} ms\n"

        return betas, [xb_gen_aa], [kpts_gen_vis], outputlogs




class ARDiffusionAction(ARDiffusion):
    
    def _setup_denoiser(self, cfg):
        self.mrepr = cfg.get('motion_repr', 'smplx_jts_locs_vel')
    
        if self.mrepr=='smplx_jts_locs_velocity_rotcont':
            # root + pose + joint locations
            self.x_dim = x_dim = 3+6 + 21*6 + 22*6
            self.n_dim_rot=6
            self.n_kpts = 22
        else:
            raise NotImplementedError('wrong mrepr')
        
        ## transformer
        assert cfg.network.get('type', 'transformerAdaLN0')=='transformerInContext', 'only support transformerInContext'

        self.controltype = cfg.network.get('controltype', 'controlnet2')
        assert self.controltype in ['finetune', 'controlnet2', 'controlnet1'], 'only support [finetune, controlnet2, controlnet1]'

        if self.controltype=='finetune':
            self.denoiser = TransformerInContext(
                x_dim,
                x_dim,
                cfg.network.h_dim,
                cfg.network.n_layer,
                cfg.network.n_head,
                dropout=cfg.network.dropout,
                n_time_embeddings = cfg.scheduler.num_train_timesteps,
                separate_condition_token=cfg.network.get('separate_condition_token', True),
                use_positional_encoding=cfg.network.get('use_positional_encoding', True),
                act_fun=cfg.network.get('act_fun', 'relu'),
            )
        elif self.controltype=='controlnet1':
            self.denoiser = TransformerInContextControlNet1(
                x_dim,
                x_dim,
                cfg.network.h_dim,
                cfg.network.n_layer,
                cfg.network.n_head,
                dropout=cfg.network.dropout,
                n_time_embeddings = cfg.scheduler.num_train_timesteps,
                separate_condition_token=cfg.network.get('separate_condition_token', True),
                use_positional_encoding=cfg.network.get('use_positional_encoding', True),
                act_fun=cfg.network.get('act_fun', 'relu'),
            )            
        elif self.controltype=='controlnet2':
            self.denoiser = TransformerInContextControlNet2(
                x_dim,
                x_dim,
                cfg.network.h_dim,
                cfg.network.n_layer,
                cfg.network.n_head,
                dropout=cfg.network.dropout,
                n_time_embeddings = cfg.scheduler.num_train_timesteps,
                separate_condition_token=cfg.network.get('separate_condition_token', True),
                use_positional_encoding=cfg.network.get('use_positional_encoding', True),
                act_fun=cfg.network.get('act_fun', 'relu'),
            )
        else:
            raise NotImplementedError



    def _setup_controller(self, cfg):
        hdim = cfg.network.h_dim
        # the existing motion seed tokenizer
        self.emb_motionseed = nn.Linear(self.x_dim, hdim)

        # setup the action embedding
        self.ctrl_action_embedding = nn.Embedding(cfg.get('maximal_action_num',10), hdim)
        

    
    def compute_ctrl_embedding(self, motionseed, action_label):
        xs_feat = self.emb_motionseed(motionseed)
        action_emb = self.ctrl_action_embedding(action_label)
        # action_feat = self.ctrl_emb(action_emb)
        
        return xs_feat, action_emb
        

    def on_fit_start(self):
        if self.controltype in ['controlnet1', 'controlnet2']:
            self.denoiser.init_weights()
        
    def forward(self,batch):
        (
            betas_all, 
            xb_all, 
            action_labels,
         ) = (
                batch["betas"], 
                batch["xb"], 
                batch["action_label"]
         )

        losses = {}
        fn_dist_fk = F.l1_loss if self.use_l1_norm_fk else F.mse_loss
        fn_dist_vel = F.l1_loss if self.use_l1_norm_vel else F.mse_loss

        # data preprcessing and compute velocity
        with torch.no_grad():
            betas = betas_all
            ## process smplx params
            if 'rotcont' in self.mrepr:
                xb_all = self.aa2rotcont(xb_all)
            
            ## motion encoding
            xb = self.canonicalization(
                betas, xb_all,
                return_transf=False
            )
            kpts = self._fwd_smplx_seq(betas, xb)
            vel = kpts[:,1:] - kpts[:,:-1]
            if self.use_metric_velocity:
                vel = vel * self.fps
            xb = xb[:,:-1]
            kpts = kpts[:,:-1]
            nb, nt = xb.shape[:2]
            
            xs = torch.cat(
                [
                    xb, 
                    kpts.reshape(nb, nt, -1),
                    vel.reshape(nb, nt, -1)
                ],dim=-1)
            
            ## add noise to the motion seed
            xs_seed = xs[:,:1].detach().clone()
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (nb,), 
                device=self.device,
                dtype=torch.int64
            )
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            xs_noise = self.noise_scheduler.add_noise(xs, torch.randn_like(xs), timesteps)
            

        xseed_emb, action_emb = self.compute_ctrl_embedding(xs_seed, action_labels)

        ## fwd pass of motion model
        xs_pred = self.denoiser(
            timesteps, 
            xs_noise,
            c_emb=xseed_emb,
            attn_mask = None,
            control_emb=action_emb,
            control_weight=1.0,
        )
        
        ## decompose prediction
        xb_pred = xs_pred[...,:-self.n_kpts*6]
        # kpts_pred = xs_pred[...,-self.n_kpts*6:-self.n_kpts*3].reshape(nb, nt, -1, 3)
        # vel_pred = xs_pred[...,-self.n_kpts*3:].reshape(nb, nt, -1, 3)
        
        # Compute the loss
        ## denoiser loss
        if self.hparams.scheduler.get('prediction_type', 'epsilon') == 'sample':
            losses['loss_simple'] = F.mse_loss(xs_pred, xs)
        else:
            raise NotImplementedError

        ## forward kinematics
        kpts_pred_fk = self._fwd_smplx_seq(betas[:,:nt], xb_pred)
        losses['loss_fk'] = fn_dist_fk(kpts_pred_fk, kpts)
        ## velocity consistency
        vel_pred_fk = kpts_pred_fk[:,1:] - kpts_pred_fk[:,:-1]
        if self.use_metric_velocity:
            vel_pred_fk = vel_pred_fk * self.fps
        losses['loss_vel'] = fn_dist_vel(vel[:,:-1], vel_pred_fk)
                
        ## total loss
        losses['loss']= self.hparams.weight_simple*losses['loss_simple']  \
            + self.hparams.weight_fk*losses['loss_fk'] \
            + self.hparams.weight_vel*losses['loss_vel']
        
        return losses




    def configure_optimizers(self):

        if self.controltype in ['controlnet1', 'controlnet2']:
            ## Initialize an empty list to hold parameters
            ft_params = []
            
            ########################################################################
            # select the cond layers
            ########################################################################
            ## Traverse all named modules in the model
            ft_params += [param for name, param in self.named_parameters() 
                                    if name.startswith('ctrl_')
                                ]

            ft_params += self.denoiser.get_ft_params()

        else:
            ft_params = self.parameters()


        ########################################################################
        # set optimizer
        ########################################################################
        optimizer = torch.optim.AdamW(
            ft_params,
            lr=self.hparams.lr
        )

        return {
            "optimizer": optimizer,
        }




    @torch.no_grad()
    def generate_perpetual_navigation(
            self,
            batch,
            n_inference_steps=10,
            nt_max = 1200,
            guidance_weight_mv=50.0,
            guidance_weight_facing=25.0,
            guidance_weight_action = 3.0,
            reproj_kpts=False,
            snap_to_ground=False,
            use_vel_perburbation=False,
            switch_on_inertialization=True,
            switch_on_control=False,

        ):
        """Generate perpetual motion with semantic action control.

        This method extends the base ARDiffusion generation with action-based control capabilities.
        Instead of trajectory following, it uses discrete action labels (e.g., jump, kick, walk)
        to guide motion generation through learned action embeddings.

        The action control is achieved through:
        - Action label embedding that conditions the diffusion process
        - Classifier-free guidance for controllable action strength
        - Optional velocity perturbations to seed specific actions
        - Optional analytical gradients for trajectory and facing control

        Args:
            batch (dict): Input batch containing:
                - betas (torch.Tensor): SMPL-X shape parameters [1, T, 16]
                - xb (torch.Tensor): Initial SMPL-X pose parameters [1, T, D]
                - ori (torch.Tensor): Desired trajectory orientation [1, T, 3]
                - action_label (torch.Tensor): Action label index [1] or [batch]

            n_inference_steps (int, optional): Number of diffusion denoising steps per primitive.
                Lower values are faster but may reduce quality. Defaults to 10.

            nt_max (int, optional): Maximum number of frames to generate in total.
                Generation stops when this limit is reached. Defaults to 1200.

            guidance_weight_mv (float, optional): Classifier guidance weight for average
                movement velocity direction. Only active if switch_on_control=True. Defaults to 50.0.

            guidance_weight_facing (float, optional): Classifier guidance weight for body
                facing direction. Only active if switch_on_control=True. Defaults to 25.0.

            guidance_weight_action (float, optional): Classifier-free guidance weight for
                action control. Higher values increase adherence to the action label.
                1.0 means no guidance, >1.0 strengthens control. Defaults to 3.0.

            reproj_kpts (bool, optional): If True, recompute keypoints via forward kinematics
                instead of transforming cached keypoints. More accurate but slower. Defaults to False.

            snap_to_ground (bool, optional): If True, projects motion to ensure contact with
                ground plane (y=0.01). Prevents floating artifacts. Defaults to False.

            use_vel_perburbation (bool, optional): If True, applies random velocity perturbations
                to the seed motion. Can help diversify action execution. Defaults to False.

            switch_on_inertialization (bool, optional): If True, applies inertialization
                (smooth blending) between consecutive motion primitives. Defaults to True.

            switch_on_control (bool, optional): If True, enables additional classifier guidance
                for trajectory and facing direction control. Defaults to False.

        Returns:
            tuple: A 5-tuple containing:
                - betas (torch.Tensor): SMPL-X shape parameters [1, 1, 16]
                - xb_gen_list (list): List with generated SMPL-X pose sequences [1, T_gen, D]
                - kpts_vis_list (list): List with keypoint+velocity visualizations [1, T_gen, N, 4]
                - outputlogs (str): Runtime statistics for each generation stage
                - xs_all (torch.Tensor): Concatenated latent representations for all primitives

        Note:
            - Uses classifier-free guidance with action label embeddings
            - Action labels are learned discrete embeddings from training data
            - Can combine action control with trajectory guidance via switch_on_control
            - Velocity perturbations apply random noise scaled by fps if metric_velocity is used
        """

        # setup data io
        (
            betas_all, 
            xb_all, 
            ori,
            action_label,
         ) = (
                batch["betas"], 
                batch["xb"], 
                batch["ori"],
                batch["action_label"],
         )
        
        # setup denoiser
        self.noise_scheduler.set_timesteps(n_inference_steps)
        nt_tw = self.hparams.data.seq_len - 1
        if not self.use_metric_velocity:
            ori = ori / self.fps

        ###################  loop of recursion #################
        tt = 0
        xb_gen = []
        kpts_gen = []
        vel_gen = []
        vel_gen_avg = []
        xs_all = []


        nb = 1    
        betas = betas_all[:,:1]

        runtimelog = {
            'canonicalization': [],
            'control_embedding': [],
            'reverse_diffusion': [],
            'post_processing': [],
        }
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

                
        while tt < nt_max:
            if tt + nt_tw > nt_max:
                break

            ################### 1. canonicalization ################
            start_event.record()
            if tt==0:
                if 'rotcont' in self.mrepr:
                    xb_all = self.aa2rotcont(xb_all)

                if snap_to_ground:
                    xb_all = self.snap_init_cond_to_ground(betas_all, xb_all)

                xb, rotmat_c, transl_c = self.canonicalization(
                    betas_all, xb_all,
                    return_transf=True
                )
                
                kpts = self._fwd_smplx_seq(betas_all, xb)
                vel_seed_c = kpts[:,1:] - kpts[:,:-1]
                if self.use_metric_velocity:
                    vel_seed_c = vel_seed_c * self.fps

                xb_seed_c = xb[:,:-1]
                kpts_seed_c = kpts[:,:-1]
                
            else:
                ## recompute keypoint locations and velocities
                xb_seed_w = xb_gen_w_[:,-1:].detach().clone()
                xb_seed_c, rotmat_c, transl_c = self.canonicalization(
                    betas.repeat(1,xb_seed_w.shape[1],1), xb_seed_w,return_transf=True)
                
                if reproj_kpts:
                    kpts_seed_c = self._fwd_smplx_seq(
                        betas.repeat(1,xb_seed_w.shape[1],1), xb_seed_c
                    )
                else:
                    kpts_seed_w = kpts_gen_w_[:,-1:].detach().clone()
                    kpts_seed_c = torch.einsum(
                        'bij,btpj->btpi', 
                        rotmat_c.permute(0,2,1), 
                        kpts_seed_w-transl_c.unsqueeze(-2)
                    )
                
                vel_seed_w = vel_gen_w_[:,-1:].detach().clone()
                vel_seed_c = torch.einsum(
                    'bij,btpj->btpi', 
                    rotmat_c.permute(0,2,1),
                    vel_seed_w
                )

           
            if use_vel_perburbation:
                # ## hit head, id=15
                # ## hit pelvis, id=0
                perturb = torch.randn_like(vel_seed_c) * 1
                vel_seed_c += perturb if self.use_metric_velocity else perturb / self.fps


            end_event.record()
            torch.cuda.synchronize()
            runtimelog['canonicalization'].append(start_event.elapsed_time(end_event))

            ################### 2. control embedding ################
            start_event.record()            
            ## obtain guidance velocity
            ori_c = torch.einsum('bij,btj->bti', rotmat_c.permute(0,2,1),ori)

            xs_seed = torch.cat([xb_seed_c, 
                                kpts_seed_c.reshape(nb, 1, -1),
                                vel_seed_c.reshape(nb, 1, -1)],
                                dim=-1)
            
            ## encode of the motion seed
            xseed_emb, action_emb = self.compute_ctrl_embedding(xs_seed, action_label)

            end_event.record()
            torch.cuda.synchronize()
            runtimelog['control_embedding'].append(start_event.elapsed_time(end_event))
            
            ################### 3. reverse diffusion ################
            start_event.record()
            input = torch.randn(nb, nt_tw, self.x_dim).to(self.device) # batch cond and uncond
            # control_weight = torch.tensor(guidance_weight_action).to(self.device)
            control_weight = torch.tensor(1.0).to(self.device)
            control_weight_0 = torch.zeros_like(control_weight)
            control_weight_all = torch.stack([control_weight, control_weight_0], dim=0)[:,None,None]

            for i, t in enumerate(self.noise_scheduler.timesteps):
                ## set time steps and the first frame
                t_tensor = t.to(self.device).unsqueeze(0).repeat(input.shape[0])          

                modeloutput_all = self.denoiser(
                    t_tensor.repeat(2), 
                    input.repeat(2,1,1),
                    c_emb=xseed_emb.repeat(2,1,1),
                    attn_mask = None,
                    control_emb = action_emb.repeat(2,1,1),
                    control_weight = control_weight_all
                )
                                          
                # modeloutput_all = self.denoiser(
                #     t_tensor.repeat(2), 
                #     input.repeat(2,1,1),
                #     c_emb=xseed_emb.repeat(2,1,1),
                #     attn_mask = self.causal_mask[:nt_tw, :nt_tw] \
                #         if self.hparams.network.is_causal else None,
                #     control_emb = action_emb.repeat(2,1,1),
                #     control_weight = control_weight_all
                # )
                modeloutput_cond, modeloutput_uncond = modeloutput_all[:nb], modeloutput_all[nb:]
                modeloutput = modeloutput_uncond + guidance_weight_action*(modeloutput_cond - modeloutput_uncond)



                # classifier guidance on the velocities
                # to speed up, we derive the analytical gradients
                ## 1) the body is supposed to move in the direction of ori_c, L = (x-ori_c)^2
                ## 2) the body is supposed to face in the direction of ori_c, the last pose facing ori_c.
                ##     namely, L = torch.sum( (cross(x/||x||, y) - ori_c/||ori_c|| )**2 )
                ## when using metric velocity, the guidance weight should be 0.01 times smaller
                if switch_on_control:
                    ## 1) moving direction
                    vels_reshaped = modeloutput[...,-3*self.n_kpts:].reshape(nb, -1, self.n_kpts, 3)
                    grad_mv_vels = 2*(vels_reshaped.mean(dim=[1,2],keepdim=True) - ori_c.unsqueeze(-2)) #[nb,1,1,3]
                    # grad_mv_vels = grad_mv_vels / (nt_tw + self.n_kpts)
                    grad_mv_vels = grad_mv_vels.repeat(1, nt_tw, self.n_kpts,1).reshape(nb, nt_tw, -1)
                    grad_mv_pad = torch.zeros_like(input[...,:-3*self.n_kpts])
                    grad_mv = torch.cat([grad_mv_pad, grad_mv_vels], dim=-1)

                    ## 2) facing direction. The last pose in the primitive is facing ori_c.
                    ## the following implementation of grad_p is identical to autodiff, already verified
                    ## guidance weight= 25-100 can give good results. When reprojecting keypoints, this weight can be smaller.
                    ### 2.1) specify input
                    ori_cn = ori_c / (ori_c.norm(dim=-1,keepdim=True) + 1e-6)
                    # ori_cn = ori / (ori.norm(dim=-1,keepdim=True) + 1e-6)
                    kpts_reshaped = modeloutput[...,-6*self.n_kpts:-3*self.n_kpts].reshape(nb, -1, self.n_kpts, 3)
                    kpts_last = kpts_reshaped[:,-1]
                    x_axis = kpts_last[:,1] - kpts_last[:,2]
                    x_axis[...,1] = 0
                    ### 2.2) calc gradient
                    x_axis_norm = x_axis.norm(dim=-1, keepdim=True)
                    y_axis = torch.tensor([[0,1,0]]).float().to(self.device).repeat(nb, 1)
                    ssmat_y = skew_symmetric_matrix(y_axis)
                    idty = torch.eye(3).to(self.device).unsqueeze(0)
                    grad_normalization = (idty - (x_axis.unsqueeze(-1) @ x_axis.unsqueeze(-2))/(x_axis_norm**2))/x_axis_norm
                    grad_p = 2*(x_axis/x_axis_norm @ ssmat_y -ori_cn) @ (-ssmat_y) @ grad_normalization
                    ### 2.3) assign gradients to the input
                    grad_kpts_all = torch.zeros_like(kpts_reshaped)
                    grad_kpts_all[:,-1:,1] = grad_p
                    grad_kpts_all[:,-1:,2] = -grad_p
                    grad_facing = torch.zeros_like(input)
                    grad_facing[...,-6*self.n_kpts:-3*self.n_kpts] = grad_kpts_all.reshape(nb, nt_tw, -1)

                    ## combine the gradients
                    guidance_grad_ori = -(guidance_weight_facing * grad_facing 
                                            + guidance_weight_mv * grad_mv)
                else:
                    guidance_grad_ori = None
                

                # step
                stepoutput = self.noise_scheduler.step(
                    modeloutput, t, input, 
                    guidance=guidance_grad_ori
                )
                input = stepoutput.prev_sample
                
            ## project to above the ground floor
            if snap_to_ground: 
                xb_pred = input[:,:,:-self.n_kpts*6]
                jts_c = self._fwd_smplx_seq(betas.repeat(1, nt_tw, 1), xb_pred)
                jts_w = torch.einsum('bij,btpj->btpi', rotmat_c, jts_c)+transl_c.unsqueeze(-2)
                heights = jts_w[...,1]
                heights_flatten = heights.view(heights.size(0), -1)
                ## compute body ground contact
                if torch.all(heights_flatten>0):
                    tpidx_flatten = heights_flatten.abs().argmin(dim=1)
                else:    
                    tpidx_flatten = heights_flatten.argmin(dim=1)
                
                dist2ground = heights_flatten[:,tpidx_flatten].item()
                ## projection
                # input[...,1] -= (dist2ground - 0.01)    
                input[...,1] -= (dist2ground - 0.01)    
            
            
            # inertialization
            if switch_on_inertialization:
                output = inertialize(xs_seed, input, omega=10.0, dt=1.0/30.0)
            else:
                output = input
            
            xs_all.append(output.detach().clone())

            end_event.record()
            torch.cuda.synchronize()
            runtimelog['reverse_diffusion'].append(start_event.elapsed_time(end_event))
            
            ################### 4. post processing ################
            start_event.record()
            output_betas = betas.repeat(1, nt_tw, 1)
            
            xb_gen_c = output[...,:-self.n_kpts*6]
            kpts_gen_c = output[...,-self.n_kpts*6:-self.n_kpts*3].reshape(nb,-1,self.n_kpts,3)
            vel_gen_c = output[...,-self.n_kpts*3:].reshape(nb,-1,self.n_kpts,3)
            
            ## transform back to the original world coordinate
            xb_gen_w_ = self.smplx_parser.update_transl_glorot_seq(
                rotmat_c, transl_c, output_betas, xb_gen_c, fwd_transf=True)
            kpts_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, kpts_gen_c) + transl_c.unsqueeze(-2)
            vel_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, vel_gen_c)
            
            xb_gen.append(xb_gen_w_[:,:-1].detach().clone())
            kpts_gen.append(kpts_gen_w_[:,:-1].detach().clone())
            vel_gen.append(vel_gen_w_[:,:-1].detach().clone())
            vel_gen_avg.append(vel_gen_w_[:,:-1].mean(dim=[1,2],keepdim=True).repeat(1,nt_tw-1,1,1).detach().clone())
            
            end_event.record()
            torch.cuda.synchronize()
            runtimelog['post_processing'].append(start_event.elapsed_time(end_event))

            tt += nt_tw-1

        # cat all sequences
        xb_gen = torch.cat(xb_gen, dim=1)
        if 'rotcont' in self.mrepr:
            xb_gen = self.rotcont2aa(xb_gen)
        kpts_gen = torch.cat(kpts_gen, dim=1)
        vel_gen = torch.cat(vel_gen, dim=1)
        vel_gen_avg = torch.cat(vel_gen_avg, dim=1)
        if not self.use_metric_velocity:
            vel_gen *= self.fps
            vel_gen_avg *= self.fps
        
        ################# generate rays: start ######################
        # Normalize the velocity vectors to get the directions
        vel_dir = vel_gen / vel_gen.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Define the step size for each point in the ray
        step_size = 0.02  # You can adjust this value as needed
        steps = torch.arange(0, 10).view(1, 1, 1, 1, -1).to(self.device)  # Shape: [1, 1, 1, 10]

        # Multiply directions by the step size and steps to create rays
        rays = kpts_gen.unsqueeze(-1) + vel_dir.unsqueeze(-1) * steps * step_size  # Shape: [b, t, p, 3, 10]

        # Rearrange the rays if necessary (e.g., [t, p, 10, 3] for compatibility with other operations)
        rays = rays.permute(0, 1, 2, 4, 3) # Now shape is [b, t, p,10, 3]
        velnorms = vel_gen.norm(dim=-1,keepdim=True).unsqueeze(-1).repeat(1,1,1,10,1)
        kpts_gen_vis = torch.cat([rays, velnorms], dim=-1)
        kpts_gen_vis = kpts_gen_vis.reshape(nb,kpts_gen.shape[1],-1,4)
        ################# generate rays: end ######################

        ################# Compute and visualize averaged velocity ######################
        # vel_gen_avg = vel_gen.mean(dim=-2,keepdim=True)
        kpts_gen_avg = kpts_gen[:,:,:1].clone()
        kpts_gen_avg[...,1] = 1.7 # place pelvis to ground
        vel_dir = vel_gen_avg / vel_gen_avg.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Define the step size for each point in the ray
        step_size = 0.05  # You can adjust this value as needed
        steps = torch.arange(0, 20).view(1, 1, 1, 1, -1).to(self.device)  # Shape: [1, 1, 1, 10]
        
        # Multiply directions by the step size and steps to create rays
        rays = kpts_gen_avg.unsqueeze(-1) + vel_dir.unsqueeze(-1) * steps * step_size  # Shape: [b, t, p, 3, 10]

        # Rearrange the rays if necessary (e.g., [t, p, 10, 3] for compatibility with other operations)
        rays = rays.permute(0, 1, 2, 4, 3) # Now shape is [b, t, p,10, 3]
        velnorms = vel_gen_avg.norm(dim=-1,keepdim=True).unsqueeze(-1).repeat(1,1,1,20,1)
        kpts_gen_avg = torch.cat([rays, velnorms], dim=-1)
        kpts_gen_avg = kpts_gen_avg.reshape(nb,kpts_gen_avg.shape[1],-1,4)
        kpts_gen_vis = torch.cat([kpts_gen_vis, kpts_gen_avg], dim=-2)


        # stats runtime
        outputlogs = f"avarage runtime to generate a motion primitive ({nt_tw} frames, {1000*nt_tw/30.:.3f} ms ): \n"
        avg = 0
        for k, v in runtimelog.items():
            outputlogs += f"-- {k}: {np.mean(v):.3f} ms\n"
            avg += np.mean(v)
        outputlogs += f"-- total: {avg:.3f} ms\n"


        return betas, [xb_gen], [kpts_gen_vis], outputlogs, torch.cat(xs_all)




class ARDiffusionSpatial(ARDiffusion):
    
    def normalize_loc(self, goal_x):
        dir_norm = goal_x.norm(dim=-1, keepdim=True)+1e-12
        goal_feat = 2.0 * (1.- (-dir_norm).exp()) * goal_x/dir_norm
        return goal_feat
    

    def _setup_denoiser(self, cfg):
        self.mrepr = cfg.get('motion_repr', 'smplx_jts_locs_vel')
    
        if self.mrepr=='smplx_jts_locs_velocity_rotcont':
            # root + pose + joint locations
            self.x_dim = x_dim = 3+6 + 21*6 + 22*6
            self.n_dim_rot=6
            self.n_kpts = 22
        else:
            raise NotImplementedError('wrong mrepr')
        
        ## transformer
        assert cfg.network.get('type', 'transformerAdaLN0')=='transformerInContext', 'only support transformerInContext'

        self.controltype = cfg.network.get('controltype', 'controlnet2')
        assert self.controltype in ['finetune', 'controlnet2', 'controlnet1'], 'only support [finetune, controlnet2, controlnet1] for adaptation'

        if self.controltype=='finetune':
            ## control signal injection
            self.denoiser = TransformerInContext(
                x_dim,
                x_dim,
                cfg.network.h_dim,
                cfg.network.n_layer,
                cfg.network.n_head,
                dropout=cfg.network.dropout,
                n_time_embeddings = cfg.scheduler.num_train_timesteps,
                separate_condition_token=cfg.network.get('separate_condition_token', True),
                use_positional_encoding=cfg.network.get('use_positional_encoding', True),
                act_fun=cfg.network.get('act_fun', 'relu'),
            )
        elif self.controltype=='controlnet1':
            ## the controlnet used in omnicontrol
            self.denoiser = TransformerInContextControlNet1(
                x_dim,
                x_dim,
                cfg.network.h_dim,
                cfg.network.n_layer,
                cfg.network.n_head,
                dropout=cfg.network.dropout,
                n_time_embeddings = cfg.scheduler.num_train_timesteps,
                separate_condition_token=cfg.network.get('separate_condition_token', True),
                use_positional_encoding=cfg.network.get('use_positional_encoding', True),
                act_fun=cfg.network.get('act_fun', 'relu'),
            )            
        elif self.controltype=='controlnet2':
            ## our controlnet
            self.denoiser = TransformerInContextControlNet2(
                x_dim,
                x_dim,
                cfg.network.h_dim,
                cfg.network.n_layer,
                cfg.network.n_head,
                dropout=cfg.network.dropout,
                n_time_embeddings = cfg.scheduler.num_train_timesteps,
                separate_condition_token=cfg.network.get('separate_condition_token', True),
                use_positional_encoding=cfg.network.get('use_positional_encoding', True),
                act_fun=cfg.network.get('act_fun', 'relu'),
            )
        else:
            raise NotImplementedError




    def _setup_controller(self, cfg):
        hdim = cfg.network.h_dim
        # the existing motion seed tokenizer
        self.emb_motionseed = nn.Linear(self.x_dim, hdim)

        # the 3D pelvis target or the 2D target?
        self.goal_type = cfg.get('goal_type', '3D')
        if self.goal_type=='3D':
            self.ctrl_target = nn.Linear(3, hdim)
        elif self.goal_type=='2D':
            self.ctrl_target = nn.Linear(2, hdim)
        else:
            raise NotImplementedError
        

    def compute_ctrl_embedding(self, motionseed, target):
        xs_feat = self.emb_motionseed(motionseed)
        target_n = self.normalize_loc(target)
        tgt_emb = self.ctrl_target(target_n)
        
        return xs_feat, tgt_emb
        

    def on_fit_start(self):
        if self.controltype in ['controlnet1', 'controlnet2']:
            self.denoiser.init_weights()
        
    def forward(self,batch):
        (
            betas_all, 
            xb_all, 
         ) = (
                batch["betas"], 
                batch["xb"], 
         )

        losses = {}
        fn_dist_fk = F.l1_loss if self.use_l1_norm_fk else F.mse_loss
        fn_dist_vel = F.l1_loss if self.use_l1_norm_vel else F.mse_loss

        # data preprcessing and compute velocity
        with torch.no_grad():
            betas = betas_all
            ## process smplx params
            if 'rotcont' in self.mrepr:
                xb_all = self.aa2rotcont(xb_all)
            
            ## motion encoding
            xb = self.canonicalization(
                betas, xb_all,
                return_transf=False
            )
            kpts = self._fwd_smplx_seq(betas, xb)
            vel = kpts[:,1:] - kpts[:,:-1]
            if self.use_metric_velocity:
                vel = vel * self.fps
            xb = xb[:,:-1]
            kpts = kpts[:,:-1]
            nb, nt = xb.shape[:2]
            
            xs = torch.cat(
                [
                    xb, 
                    kpts.reshape(nb, nt, -1),
                    vel.reshape(nb, nt, -1)
                ],dim=-1)
            
            ## add noise to the motion seed
            xs_seed = xs[:,:1].detach().clone()
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (nb,), 
                device=self.device,
                dtype=torch.int64
            )
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            xs_noise = self.noise_scheduler.add_noise(xs, torch.randn_like(xs), timesteps)
            
            # obtain the target pelvis location
            target = kpts[:,-1:,0].detach().clone()
            if self.goal_type=='2D':
                target = target[...,[0,2]]
            
        xseed_emb, tgt_emb = self.compute_ctrl_embedding(xs_seed, target)

        ## fwd pass of motion model
        xs_pred = self.denoiser(
            timesteps, 
            xs_noise,
            c_emb=xseed_emb,
            attn_mask = None,
            control_emb=tgt_emb,
            control_weight=1.0
        )
        
        ## decompose prediction
        xb_pred = xs_pred[...,:-self.n_kpts*6]
        
        # Compute the loss
        ## denoiser loss
        if self.hparams.scheduler.get('prediction_type', 'epsilon') == 'sample':
            losses['loss_simple'] = F.mse_loss(xs_pred, xs)
        else:
            raise NotImplementedError

        ## forward kinematics
        kpts_pred_fk = self._fwd_smplx_seq(betas[:,:nt], xb_pred)
        losses['loss_fk'] = fn_dist_fk(kpts_pred_fk, kpts)
        ## velocity consistency
        vel_pred_fk = kpts_pred_fk[:,1:] - kpts_pred_fk[:,:-1]
        if self.use_metric_velocity:
            vel_pred_fk = vel_pred_fk * self.fps
        losses['loss_vel'] = fn_dist_vel(vel[:,:-1], vel_pred_fk)
                
        ## total loss
        losses['loss']= self.hparams.weight_simple*losses['loss_simple']  \
            + self.hparams.weight_fk*losses['loss_fk'] \
            + self.hparams.weight_vel*losses['loss_vel']
        
        return losses







    def configure_optimizers(self):

        if self.controltype in ['controlnet1', 'controlnet2']:
            ## Initialize an empty list to hold parameters
            ft_params = []
            
            ########################################################################
            # select the cond layers
            ########################################################################
            ## Traverse all named modules in the model
            ft_params += [param for name, param in self.named_parameters() 
                                    if name.startswith('ctrl_')
                                ]

            ft_params += self.denoiser.get_ft_params()

        else:
            ft_params = self.parameters()


        ########################################################################
        # set optimizer
        ########################################################################
        optimizer = torch.optim.AdamW(
            ft_params,
            lr=self.hparams.lr,
        )

        return {
            "optimizer": optimizer,
        }



    @torch.no_grad()
    def generate_perpetual_navigation(
            self,
            batch,
            n_inference_steps=10,
            nt_max = 1200,
            guidance_weight_facing=25.0,
            guidance_weight_action = 1.0,
            reproj_kpts=False,
            snap_to_ground=False,
            switch_on_inertialization=True,
            switch_on_control=False,
        ):
        """Generate perpetual motion following a spatial target location.

        This method extends the base ARDiffusion generation with spatial control capabilities.
        Instead of velocity-based navigation, it uses a target pelvis location to guide motion
        generation through learned control embeddings (via ControlNet or fine-tuning).

        The spatial control is achieved through:
        - Target location embedding that guides the diffusion process
        - Classifier-free guidance for controllable generation strength
        - Optional analytical gradients for facing direction control

        Args:
            batch (dict): Input batch containing:
                - betas (torch.Tensor): SMPL-X shape parameters [1, T, 16]
                - xb (torch.Tensor): Initial SMPL-X pose parameters [1, T, D]
                - ori (torch.Tensor): Desired trajectory orientation [1, T, 3]

            n_inference_steps (int, optional): Number of diffusion denoising steps per primitive.
                Lower values are faster but may reduce quality. Defaults to 10.

            nt_max (int, optional): Maximum number of frames to generate in total.
                Generation stops when this limit is reached. Defaults to 1200.

            guidance_weight_facing (float, optional): Classifier guidance weight for body
                facing direction (analytical gradient-based). Only active if switch_on_control=True.
                Defaults to 25.0.

            guidance_weight_action (float, optional): Classifier-free guidance weight for
                spatial target reaching. Higher values increase adherence to target location.
                1.0 means no guidance, >1.0 strengthens control. Defaults to 1.0.

            reproj_kpts (bool, optional): If True, recompute keypoints via forward kinematics
                instead of transforming cached keypoints. More accurate but slower. Defaults to False.

            snap_to_ground (bool, optional): If True, projects motion to ensure contact with
                ground plane (y=0.01). Prevents floating artifacts. Defaults to False.

            switch_on_inertialization (bool, optional): If True, applies inertialization
                (smooth blending) between consecutive motion primitives. Defaults to True.

            switch_on_control (bool, optional): If True, enables additional classifier guidance
                for facing direction control. Defaults to False.

        Returns:
            tuple: A 4-tuple containing:
                - betas (torch.Tensor): SMPL-X shape parameters [1, 1, 16]
                - xb_gen_list (list): List with generated SMPL-X pose sequences [1, T_gen, D]
                - kpts_vis_list (list): List with keypoint+velocity visualizations [1, T_gen, N, 4]
                - outputlogs (str): Runtime statistics for each generation stage

        Note:
            - Uses classifier-free guidance with target location embeddings
            - Target location is computed as: current_position + trajectory_offset
            - Supports both 2D (xz-plane) and 3D target specifications via goal_type
            - ControlNet-based models (controltype='controlnet1/2') only update control parameters
        """

        # setup data io
        (
            betas_all, 
            xb_all, 
            ori_o,
         ) = (
                batch["betas"], 
                batch["xb"], 
                batch["ori"],
         )
        
        # setup denoiser
        self.noise_scheduler.set_timesteps(n_inference_steps)
        nt_tw = self.hparams.data.seq_len - 1
        if not self.use_metric_velocity:
            ori = ori_o / self.fps


                

        ###################  loop of recursion #################
        tt = 0
        xb_gen = []
        kpts_gen = []
        vel_gen = []
        vel_gen_avg = []

        nb = 1    
        betas = betas_all[:,:1]

        runtimelog = {
            'canonicalization': [],
            'control_embedding': [],
            'reverse_diffusion': [],
            'post_processing': [],
        }
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        
        while tt < nt_max:
            if tt + nt_tw > nt_max:
                break

            ################### 1. canonicalization ################
            start_event.record()
            if tt==0:
                if 'rotcont' in self.mrepr:
                    xb_all = self.aa2rotcont(xb_all)

                if snap_to_ground:
                    xb_all = self.snap_init_cond_to_ground(betas_all, xb_all)

                xb, rotmat_c, transl_c = self.canonicalization(
                    betas_all, xb_all,
                    return_transf=True
                )

                kpts = self._fwd_smplx_seq(betas_all, xb)
                vel_seed_c = kpts[:,1:] - kpts[:,:-1]
                if self.use_metric_velocity:
                    vel_seed_c = vel_seed_c * self.fps

                xb_seed_c = xb[:,:-1]
                kpts_seed_c = kpts[:,:-1]

                # specify the target location
                tgt_pelvis_loc =  transl_c.detach().clone() +  ori_o
        
                
            else:
                ## recompute keypoint locations and velocities
                xb_seed_w = xb_gen_w_[:,-1:].detach().clone()
                xb_seed_c, rotmat_c, transl_c = self.canonicalization(
                    betas.repeat(1,xb_seed_w.shape[1],1), xb_seed_w,return_transf=True)
                
                if reproj_kpts:
                    kpts_seed_c = self._fwd_smplx_seq(
                        betas.repeat(1,xb_seed_w.shape[1],1), xb_seed_c
                    )
                else:
                    kpts_seed_w = kpts_gen_w_[:,-1:].detach().clone()
                    kpts_seed_c = torch.einsum(
                        'bij,btpj->btpi', 
                        rotmat_c.permute(0,2,1), 
                        kpts_seed_w-transl_c.unsqueeze(-2)
                    )
                
                vel_seed_w = vel_gen_w_[:,-1:].detach().clone()
                vel_seed_c = torch.einsum(
                    'bij,btpj->btpi', 
                    rotmat_c.permute(0,2,1),
                    vel_seed_w
                )


            end_event.record()
            torch.cuda.synchronize()
            runtimelog['canonicalization'].append(start_event.elapsed_time(end_event))

            ################### 2. control embedding ################
            start_event.record()            
            ## obtain guidance velocity
            ori_c = torch.einsum('bij,btj->bti', rotmat_c.permute(0,2,1),ori)
            target_c = torch.einsum(
                    'bij,btj->bti', 
                    rotmat_c.permute(0,2,1),tgt_pelvis_loc-transl_c)
            if self.goal_type=='2D':
                target_c = target_c[...,[0,2]]
            
            
            xs_seed = torch.cat([xb_seed_c, 
                                kpts_seed_c.reshape(nb, 1, -1),
                                vel_seed_c.reshape(nb, 1, -1)],
                                dim=-1)
            
            ## encode of the motion seed
            xseed_emb, tgt_emb = self.compute_ctrl_embedding(xs_seed, target_c)

            end_event.record()
            torch.cuda.synchronize()
            runtimelog['control_embedding'].append(start_event.elapsed_time(end_event))
            
            ################### 3. reverse diffusion ################
            start_event.record()
            input = torch.randn(nb, nt_tw, self.x_dim).to(self.device) # batch cond and uncond
            # control_weight = torch.tensor(guidance_weight_action).to(self.device)
            control_weight = torch.tensor(1.0).to(self.device)
            control_weight_0 = torch.zeros_like(control_weight)
            control_weight_all = torch.stack([control_weight, control_weight_0], dim=0)[:,None,None]

            for i, t in enumerate(self.noise_scheduler.timesteps):
                ## set time steps and the first frame
                t_tensor = t.to(self.device).unsqueeze(0).repeat(input.shape[0])          

                modeloutput_all = self.denoiser(
                    t_tensor.repeat(2), 
                    input.repeat(2,1,1),
                    c_emb=xseed_emb.repeat(2,1,1),
                    attn_mask = None,
                    control_emb = tgt_emb.repeat(2,1,1),
                    control_weight = control_weight_all
                )
                modeloutput_cond, modeloutput_uncond = modeloutput_all[:nb], modeloutput_all[nb:]
                modeloutput = modeloutput_uncond + guidance_weight_action*(modeloutput_cond - modeloutput_uncond)



                # classifier guidance on the velocities
                # to speed up, we derive the analytical gradients
                ## 1) the body is supposed to move in the direction of ori_c, L = (x-ori_c)^2
                ## 2) the body is supposed to face in the direction of ori_c, the last pose facing ori_c.
                ##     namely, L = torch.sum( (cross(x/||x||, y) - ori_c/||ori_c|| )**2 )
                ## when using metric velocity, the guidance weight should be 0.01 times smaller
                if switch_on_control:
                    # ## 1) moving direction
                    # vels_reshaped = modeloutput[...,-3*self.n_kpts:].reshape(nb, -1, self.n_kpts, 3)
                    # grad_mv_vels = 2*(vels_reshaped.mean(dim=[1,2],keepdim=True) - ori_c.unsqueeze(-2)) #[nb,1,1,3]
                    # # grad_mv_vels = grad_mv_vels / (nt_tw + self.n_kpts)
                    # grad_mv_vels = grad_mv_vels.repeat(1, nt_tw, self.n_kpts,1).reshape(nb, nt_tw, -1)
                    # grad_mv_pad = torch.zeros_like(input[...,:-3*self.n_kpts])
                    # grad_mv = torch.cat([grad_mv_pad, grad_mv_vels], dim=-1)

                    ## 2) facing direction. The last pose in the primitive is facing ori_c.
                    ## the following implementation of grad_p is identical to autodiff, already verified
                    ## guidance weight= 25-100 can give good results. When reprojecting keypoints, this weight can be smaller.
                    ### 2.1) specify input
                    ori_cn = ori_c / (ori_c.norm(dim=-1,keepdim=True) + 1e-6)
                    # ori_cn = ori / (ori.norm(dim=-1,keepdim=True) + 1e-6)
                    kpts_reshaped = modeloutput[...,-6*self.n_kpts:-3*self.n_kpts].reshape(nb, -1, self.n_kpts, 3)
                    kpts_last = kpts_reshaped[:,-1]
                    x_axis = kpts_last[:,1] - kpts_last[:,2]
                    x_axis[...,1] = 0
                    ### 2.2) calc gradient
                    x_axis_norm = x_axis.norm(dim=-1, keepdim=True)
                    y_axis = torch.tensor([[0,1,0]]).float().to(self.device).repeat(nb, 1)
                    ssmat_y = skew_symmetric_matrix(y_axis)
                    idty = torch.eye(3).to(self.device).unsqueeze(0)
                    grad_normalization = (idty - (x_axis.unsqueeze(-1) @ x_axis.unsqueeze(-2))/(x_axis_norm**2))/x_axis_norm
                    grad_p = 2*(x_axis/x_axis_norm @ ssmat_y -ori_cn) @ (-ssmat_y) @ grad_normalization
                    ### 2.3) assign gradients to the input
                    grad_kpts_all = torch.zeros_like(kpts_reshaped)
                    grad_kpts_all[:,-1:,1] = grad_p
                    grad_kpts_all[:,-1:,2] = -grad_p
                    grad_facing = torch.zeros_like(input)
                    grad_facing[...,-6*self.n_kpts:-3*self.n_kpts] = grad_kpts_all.reshape(nb, nt_tw, -1)

                    ## combine the gradients
                    guidance_grad_ori = -(guidance_weight_facing * grad_facing)
                else:
                    guidance_grad_ori = None
                

                # step
                stepoutput = self.noise_scheduler.step(
                    modeloutput, t, input, 
                    guidance=guidance_grad_ori
                )
                input = stepoutput.prev_sample
                
            ## project to above the ground floor
            if snap_to_ground: 
                xb_pred = input[:,:,:-self.n_kpts*6]
                jts_c = self._fwd_smplx_seq(betas.repeat(1, nt_tw, 1), xb_pred)
                jts_w = torch.einsum('bij,btpj->btpi', rotmat_c, jts_c)+transl_c.unsqueeze(-2)
                heights = jts_w[...,1]
                heights_flatten = heights.view(heights.size(0), -1)
                ## compute body ground contact
                if torch.all(heights_flatten>0):
                    tpidx_flatten = heights_flatten.abs().argmin(dim=1)
                else:    
                    tpidx_flatten = heights_flatten.argmin(dim=1)
                
                dist2ground = heights_flatten[:,tpidx_flatten].item()
                ## projection
                # input[...,1] -= (dist2ground - 0.01)    
                input[...,1] -= (dist2ground - 0.01)    
            
            
            # inertialization
            if switch_on_inertialization:
                output = inertialize(xs_seed, input, omega=10.0, dt=1.0/30.0)
            else:
                output = input
            
            
            end_event.record()
            torch.cuda.synchronize()
            runtimelog['reverse_diffusion'].append(start_event.elapsed_time(end_event))
            
            ################### 4. post processing ################
            start_event.record()
            output_betas = betas.repeat(1, nt_tw, 1)
            
            xb_gen_c = output[...,:-self.n_kpts*6]
            kpts_gen_c = output[...,-self.n_kpts*6:-self.n_kpts*3].reshape(nb,-1,self.n_kpts,3)
            vel_gen_c = output[...,-self.n_kpts*3:].reshape(nb,-1,self.n_kpts,3)
            
            ## transform back to the original world coordinate
            xb_gen_w_ = self.smplx_parser.update_transl_glorot_seq(
                rotmat_c, transl_c, output_betas, xb_gen_c, fwd_transf=True)
            kpts_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, kpts_gen_c) + transl_c.unsqueeze(-2)
            vel_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, vel_gen_c)
            
            xb_gen.append(xb_gen_w_[:,:-1].detach().clone())
            kpts_gen.append(kpts_gen_w_[:,:-1].detach().clone())
            vel_gen.append(vel_gen_w_[:,:-1].detach().clone())
            vel_gen_avg.append(vel_gen_w_[:,:-1].mean(dim=[1,2],keepdim=True).repeat(1,nt_tw-1,1,1).detach().clone())
            
            end_event.record()
            torch.cuda.synchronize()
            runtimelog['post_processing'].append(start_event.elapsed_time(end_event))

            tt += nt_tw-1

        # cat all sequences
        xb_gen = torch.cat(xb_gen, dim=1)
        if 'rotcont' in self.mrepr:
            xb_gen = self.rotcont2aa(xb_gen)
        kpts_gen = torch.cat(kpts_gen, dim=1)
        vel_gen = torch.cat(vel_gen, dim=1)
        vel_gen_avg = torch.cat(vel_gen_avg, dim=1)
        if not self.use_metric_velocity:
            vel_gen *= self.fps
            vel_gen_avg *= self.fps
        
        ################# generate rays: start ######################
        # Normalize the velocity vectors to get the directions
        vel_dir = vel_gen / vel_gen.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Define the step size for each point in the ray
        step_size = 0.02  # You can adjust this value as needed
        steps = torch.arange(0, 10).view(1, 1, 1, 1, -1).to(self.device)  # Shape: [1, 1, 1, 10]

        # Multiply directions by the step size and steps to create rays
        rays = kpts_gen.unsqueeze(-1) + vel_dir.unsqueeze(-1) * steps * step_size  # Shape: [b, t, p, 3, 10]

        # Rearrange the rays if necessary (e.g., [t, p, 10, 3] for compatibility with other operations)
        rays = rays.permute(0, 1, 2, 4, 3) # Now shape is [b, t, p,10, 3]
        velnorms = vel_gen.norm(dim=-1,keepdim=True).unsqueeze(-1).repeat(1,1,1,10,1)
        kpts_gen_vis = torch.cat([rays, velnorms], dim=-1)
        kpts_gen_vis = kpts_gen_vis.reshape(nb,kpts_gen.shape[1],-1,4)
        ################# generate rays: end ######################

        ################# Compute and visualize averaged velocity ######################
        # vel_gen_avg = vel_gen.mean(dim=-2,keepdim=True)
        kpts_gen_avg = kpts_gen[:,:,:1].clone()
        kpts_gen_avg[...,1] = 1.7 # place pelvis to ground
        vel_dir = vel_gen_avg / vel_gen_avg.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Define the step size for each point in the ray
        step_size = 0.05  # You can adjust this value as needed
        steps = torch.arange(0, 20).view(1, 1, 1, 1, -1).to(self.device)  # Shape: [1, 1, 1, 10]
        
        # Multiply directions by the step size and steps to create rays
        rays = kpts_gen_avg.unsqueeze(-1) + vel_dir.unsqueeze(-1) * steps * step_size  # Shape: [b, t, p, 3, 10]

        # Rearrange the rays if necessary (e.g., [t, p, 10, 3] for compatibility with other operations)
        rays = rays.permute(0, 1, 2, 4, 3) # Now shape is [b, t, p,10, 3]
        velnorms = vel_gen_avg.norm(dim=-1,keepdim=True).unsqueeze(-1).repeat(1,1,1,20,1)
        kpts_gen_avg = torch.cat([rays, velnorms], dim=-1)
        kpts_gen_avg = kpts_gen_avg.reshape(nb,kpts_gen_avg.shape[1],-1,4)
        kpts_gen_vis = torch.cat([kpts_gen_vis, kpts_gen_avg], dim=-2)


        # stats runtime
        outputlogs = f"avarage runtime to generate a motion primitive ({nt_tw} frames, {1000*nt_tw/30.:.3f} ms ): \n"
        avg = 0
        for k, v in runtimelog.items():
            outputlogs += f"-- {k}: {np.mean(v):.3f} ms\n"
            avg += np.mean(v)
        outputlogs += f"-- total: {avg:.3f} ms\n"


        return betas, [xb_gen], [kpts_gen_vis], outputlogs


    @torch.no_grad()
    def generate_perpetual_navigation_ue(
            self,
            batch,
            n_inference_steps=10,
            nt_max = 1200,
            guidance_weight_facing=25.0,
            guidance_weight_action = 1.0,
            snap_to_ground=False,
            switch_on_inertialization=True,
            switch_on_control=False,
            target_distance_speed = 1.0,
            poking_vector = None,
            poking_joint = None,
            movement_vector = None,
            use_vel_to_jump = False, 
        ):
        """Generate perpetual motion following a spatial target location.

        This method extends the base ARDiffusion generation with spatial control capabilities.
        Instead of velocity-based navigation, it uses a target pelvis location to guide motion
        generation through learned control embeddings (via ControlNet or fine-tuning).

        The spatial control is achieved through:
        - Target location embedding that guides the diffusion process
        - Classifier-free guidance for controllable generation strength
        - Optional analytical gradients for facing direction control

        Args:
            batch (dict): Input batch containing:
                - betas (torch.Tensor): SMPL-X shape parameters [1, T, 16]
                - xb (torch.Tensor): Initial SMPL-X pose parameters [1, T, D]
                - ori (torch.Tensor): Desired trajectory orientation [1, T, 3]

            n_inference_steps (int, optional): Number of diffusion denoising steps per primitive.
                Lower values are faster but may reduce quality. Defaults to 10.

            nt_max (int, optional): Maximum number of frames to generate in total.
                Generation stops when this limit is reached. Defaults to 1200.

            guidance_weight_facing (float, optional): Classifier guidance weight for body
                facing direction (analytical gradient-based). Only active if switch_on_control=True.
                Defaults to 25.0.

            guidance_weight_action (float, optional): Classifier-free guidance weight for
                spatial target reaching. Higher values increase adherence to target location.
                1.0 means no guidance, >1.0 strengthens control. Defaults to 1.0.

            reproj_kpts (bool, optional): If True, recompute keypoints via forward kinematics
                instead of transforming cached keypoints. More accurate but slower. Defaults to False.

            snap_to_ground (bool, optional): If True, projects motion to ensure contact with
                ground plane (y=0.01). Prevents floating artifacts. Defaults to False.

            switch_on_inertialization (bool, optional): If True, applies inertialization
                (smooth blending) between consecutive motion primitives. Defaults to True.

            switch_on_control (bool, optional): If True, enables additional classifier guidance
                for facing direction control. Defaults to False.

        Returns:
            tuple: A 4-tuple containing:
                - betas (torch.Tensor): SMPL-X shape parameters [1, 1, 16]
                - xb_gen_list (list): List with generated SMPL-X pose sequences [1, T_gen, D]
                - kpts_vis_list (list): List with keypoint+velocity visualizations [1, T_gen, N, 4]
                - outputlogs (str): Runtime statistics for each generation stage

        Note:
            - Uses classifier-free guidance with target location embeddings
            - Target location is computed as: current_position + trajectory_offset
            - Supports both 2D (xz-plane) and 3D target specifications via goal_type
            - ControlNet-based models (controltype='controlnet1/2') only update control parameters
        """

        # setup data io
        (
            betas_all, 
            xb_all, 
            ori_o,
         ) = (
                batch["betas"], 
                batch["xb"], 
                batch["ori"],
         )
        
        # setup denoiser
        self.noise_scheduler.set_timesteps(n_inference_steps)
        nt_tw = self.hparams.data.seq_len - 1
        if not self.use_metric_velocity:
            ori = ori_o / self.fps

                
        ###################  loop of recursion #################
        tt = 0
        xb_gen = []
        kpts_gen = []
        vel_gen = []
        vel_gen_avg = []

        nb = 1    
        betas = betas_all[:,:1]

        
        while tt < nt_max:
            if tt + nt_tw > nt_max:
                break

            ################### 1. canonicalization ################
            if tt==0:
                if 'rotcont' in self.mrepr:
                    xb_all = self.aa2rotcont(xb_all)

                if snap_to_ground:
                    xb_all = self.snap_init_cond_to_ground(betas_all, xb_all)

                xb, rotmat_c, transl_c = self.canonicalization(
                    betas_all, xb_all,
                    return_transf=True
                )

                kpts = self._fwd_smplx_seq(betas_all, xb)
                vel_seed_c = kpts[:,1:] - kpts[:,:-1]
                if self.use_metric_velocity:
                    vel_seed_c = vel_seed_c * self.fps

                xb_seed_c = xb[:,:-1]
                kpts_seed_c = kpts[:,:-1]

                # specify the target location
                movement_normalized = movement_vector / torch.norm(movement_vector,dim=-1,keepdim=True)
                tgt_pelvis_loc = transl_c + movement_normalized * target_distance_speed

                if use_vel_to_jump:
                    up_impulse = torch.zeros_like(vel_seed_c)
                    up_impulse[...,1] = 0.1
                    vel_seed_c += up_impulse
                    use_vel_to_jump = False

        
            else:
                raise ValueError("in the UE demo, we only generate an atomic action each time.")


            if poking_vector is not None and poking_joint is not None:
                ## hit head, id=15  
                ## hit pelvis, id=0
                vel_seed_c[:, -1, poking_joint, :] -= torch.tensor(poking_vector).float().to(self.device)


            ################### 2. control embedding ################
            ## obtain guidance velocity
            ori_c = torch.einsum('bij,btj->bti', rotmat_c.permute(0,2,1),ori)
            target_c = torch.einsum(
                    'bij,btj->bti', 
                    rotmat_c.permute(0,2,1),tgt_pelvis_loc-transl_c)
            if self.goal_type=='2D':
                target_c = target_c[...,[0,2]]
            
            xs_seed = torch.cat([xb_seed_c, 
                                kpts_seed_c.reshape(nb, 1, -1),
                                vel_seed_c.reshape(nb, 1, -1)],
                                dim=-1)
            
            ## encode of the motion seed
            xseed_emb, tgt_emb = self.compute_ctrl_embedding(xs_seed, target_c)

            
            ################### 3. reverse diffusion ################
            input = torch.randn(nb, nt_tw, self.x_dim).to(self.device) # batch cond and uncond
            # control_weight = torch.tensor(guidance_weight_action).to(self.device)
            control_weight = torch.tensor(1.0).to(self.device)
            control_weight_0 = torch.zeros_like(control_weight)
            control_weight_all = torch.stack([control_weight, control_weight_0], dim=0)[:,None,None]

            for i, t in enumerate(self.noise_scheduler.timesteps):
                ## set time steps and the first frame
                t_tensor = t.to(self.device).unsqueeze(0).repeat(input.shape[0])          

                modeloutput_all = self.denoiser(
                    t_tensor.repeat(2), 
                    input.repeat(2,1,1),
                    c_emb=xseed_emb.repeat(2,1,1),
                    attn_mask = None,
                    control_emb = tgt_emb.repeat(2,1,1),
                    control_weight = control_weight_all
                )
                modeloutput_cond, modeloutput_uncond = modeloutput_all[:nb], modeloutput_all[nb:]
                modeloutput = modeloutput_uncond + guidance_weight_action*(modeloutput_cond - modeloutput_uncond)


                # classifier guidance on the velocities
                # to speed up, we derive the analytical gradients
                ## 1) the body is supposed to move in the direction of ori_c, L = (x-ori_c)^2
                ## 2) the body is supposed to face in the direction of ori_c, the last pose facing ori_c.
                ##     namely, L = torch.sum( (cross(x/||x||, y) - ori_c/||ori_c|| )**2 )
                ## when using metric velocity, the guidance weight should be 0.01 times smaller
                if switch_on_control:
                    # ## 1) moving direction
                    # vels_reshaped = modeloutput[...,-3*self.n_kpts:].reshape(nb, -1, self.n_kpts, 3)
                    # grad_mv_vels = 2*(vels_reshaped.mean(dim=[1,2],keepdim=True) - ori_c.unsqueeze(-2)) #[nb,1,1,3]
                    # # grad_mv_vels = grad_mv_vels / (nt_tw + self.n_kpts)
                    # grad_mv_vels = grad_mv_vels.repeat(1, nt_tw, self.n_kpts,1).reshape(nb, nt_tw, -1)
                    # grad_mv_pad = torch.zeros_like(input[...,:-3*self.n_kpts])
                    # grad_mv = torch.cat([grad_mv_pad, grad_mv_vels], dim=-1)

                    ## 2) facing direction. The last pose in the primitive is facing ori_c.
                    ## the following implementation of grad_p is identical to autodiff, already verified
                    ## guidance weight= 25-100 can give good results. When reprojecting keypoints, this weight can be smaller.
                    ### 2.1) specify input
                    ori_cn = ori_c / (ori_c.norm(dim=-1,keepdim=True) + 1e-6)
                    # ori_cn = ori / (ori.norm(dim=-1,keepdim=True) + 1e-6)
                    kpts_reshaped = modeloutput[...,-6*self.n_kpts:-3*self.n_kpts].reshape(nb, -1, self.n_kpts, 3)
                    kpts_last = kpts_reshaped[:,-1]
                    x_axis = kpts_last[:,1] - kpts_last[:,2]
                    x_axis[...,1] = 0
                    ### 2.2) calc gradient
                    x_axis_norm = x_axis.norm(dim=-1, keepdim=True)
                    y_axis = torch.tensor([[0,1,0]]).float().to(self.device).repeat(nb, 1)
                    ssmat_y = skew_symmetric_matrix(y_axis)
                    idty = torch.eye(3).to(self.device).unsqueeze(0)
                    grad_normalization = (idty - (x_axis.unsqueeze(-1) @ x_axis.unsqueeze(-2))/(x_axis_norm**2))/x_axis_norm
                    grad_p = 2*(x_axis/x_axis_norm @ ssmat_y -ori_cn) @ (-ssmat_y) @ grad_normalization
                    ### 2.3) assign gradients to the input
                    grad_kpts_all = torch.zeros_like(kpts_reshaped)
                    grad_kpts_all[:,-1:,1] = grad_p
                    grad_kpts_all[:,-1:,2] = -grad_p
                    grad_facing = torch.zeros_like(input)
                    grad_facing[...,-6*self.n_kpts:-3*self.n_kpts] = grad_kpts_all.reshape(nb, nt_tw, -1)

                    ## combine the gradients
                    guidance_grad_ori = -(guidance_weight_facing * grad_facing)
                else:
                    guidance_grad_ori = None
                

                # step
                stepoutput = self.noise_scheduler.step(
                    modeloutput, t, input, 
                    guidance=guidance_grad_ori
                )
                input = stepoutput.prev_sample
                
            ## project to above the ground floor
            if snap_to_ground: 
                xb_pred = input[:,:,:-self.n_kpts*6]
                jts_c = self._fwd_smplx_seq(betas.repeat(1, nt_tw, 1), xb_pred)
                jts_w = torch.einsum('bij,btpj->btpi', rotmat_c, jts_c)+transl_c.unsqueeze(-2)
                heights = jts_w[...,1]
                heights_flatten = heights.view(heights.size(0), -1)
                ## compute body ground contact
                if torch.all(heights_flatten>0):
                    tpidx_flatten = heights_flatten.abs().argmin(dim=1)
                else:    
                    tpidx_flatten = heights_flatten.argmin(dim=1)
                
                dist2ground = heights_flatten[:,tpidx_flatten].item()
                ## projection
                # input[...,1] -= (dist2ground - 0.01)    
                input[...,1] -= (dist2ground - 0.01)    
            
            
            # inertialization
            if switch_on_inertialization:
                output = inertialize(xs_seed, input, omega=10.0, dt=1.0/30.0)
            else:
                output = input
            
            
            ################### 4. post processing ################
            output_betas = betas.repeat(1, nt_tw, 1)
            
            xb_gen_c = output[...,:-self.n_kpts*6]
            kpts_gen_c = output[...,-self.n_kpts*6:-self.n_kpts*3].reshape(nb,-1,self.n_kpts,3)
            vel_gen_c = output[...,-self.n_kpts*3:].reshape(nb,-1,self.n_kpts,3)
            
            ## transform back to the original world coordinate
            xb_gen_w_ = self.smplx_parser.update_transl_glorot_seq(
                rotmat_c, transl_c, output_betas, xb_gen_c, fwd_transf=True)
            kpts_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, kpts_gen_c) + transl_c.unsqueeze(-2)
            vel_gen_w_ = torch.einsum('bij,btpj->btpi', rotmat_c, vel_gen_c)
            
            xb_gen.append(xb_gen_w_[:,:-1].detach().clone())
            kpts_gen.append(kpts_gen_w_[:,:-1].detach().clone())
            vel_gen.append(vel_gen_w_[:,:-1].detach().clone())
            vel_gen_avg.append(vel_gen_w_[:,:-1].mean(dim=[1,2],keepdim=True).repeat(1,nt_tw-1,1,1).detach().clone())
            
            tt += nt_tw-1

        # cat all sequences
        xb_gen = torch.cat(xb_gen, dim=1)
        if 'rotcont' in self.mrepr:
            xb_gen = self.rotcont2aa(xb_gen)
        kpts_gen = torch.cat(kpts_gen, dim=1)
        vel_gen = torch.cat(vel_gen, dim=1)
        vel_gen_avg = torch.cat(vel_gen_avg, dim=1)
        if not self.use_metric_velocity:
            vel_gen *= self.fps
            vel_gen_avg *= self.fps
        
        

        return betas, [xb_gen], [kpts_gen]


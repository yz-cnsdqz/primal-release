from lightning import LightningModule
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.mop_repr import SMPLXParserRotcont, RotConverter
from .base_models import *





class MotionCLIP(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self._setup_smplx(cfg)
        self._setup_motionencoder(cfg)
        self._setup_textencoder(cfg)
        self._setup_additional(cfg)



    def _setup_smplx(self, cfg):
        """
        Setup SMPL-X parser for motion representation.

        Uses MODEL_REGISTRY_PATH environment variable for model path if available,
        otherwise falls back to config value.
        """
        # Use MODEL_REGISTRY_PATH environment variable if available, otherwise fall back to config
        model_reg_path = os.getenv('MODEL_REGISTRY_PATH', getattr(cfg.data, 'model_reg_path', None))
        if model_reg_path is None:
            raise ValueError("MODEL_REGISTRY_PATH environment variable not set and model_reg_path not found in config")

        self.smplx_parser = SMPLXParserRotcont(osp.join(model_reg_path,"models/SMPLX/neutral/SMPLX_neutral.npz"),
                                        "primal/utils/SSM2.json",
                                        num_betas=16)
        
        self.smplx_parser.eval()

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
        

    def _setup_motionencoder(self, cfg):
        # define motion representation
        self.mrepr = cfg.get('motion_repr', 'smplx_jts_locs_velocity_rotcont')
        assert self.mrepr== 'smplx_jts_locs_velocity_rotcont', 'only support smplx_jts_locs_velocity_rotcont for now'
        # root + pose + joint locations
        self.x_dim = x_dim = 3+6 + 21*6 + 22*6
        self.n_dim_rot=6
        self.n_kpts = 22
        
        # define network
        h_dim = cfg.network.h_dim
        self.pos_embedding = PositionalEncoder(h_dim, dropout=0)
        

        # transformer
        self.in_fc = nn.Linear(x_dim, h_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=cfg.network.n_head,
                                               batch_first=True, 
                                               dropout=cfg.network.dropout,
                                               norm_first=True,
                                               activation=F.silu
                                               )
        self.tf_encoder = nn.TransformerEncoder(enc_layer, 
                                                num_layers=cfg.network.n_layer,
                                                enable_nested_tensor=False)
        
    def _setup_textencoder(self, cfg):
        h_dim = cfg.network.h_dim
        self.text_encoder = nn.Embedding(cfg.n_vocab, h_dim)  
        


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


    def encode_motion(self, sample):

        # 2. encoding motion patterns
        hxt = self.in_fc(sample) 

        # 3. apply positional encoding
        hxt = self.pos_embedding(hxt)
        
        # 4. transformer blocks
        hxt = self.tf_encoder(hxt)

        return hxt[:,-1] #[b,d]


    def encode_text(self, action_label):
        text_emb = self.text_encoder(action_label)
        return text_emb


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
        
        # encode motion features
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
            xb = xb[:,:-1]
            kpts = kpts[:,:-1]
            nb, nt = xb.shape[:2]
            
            xs = torch.cat(
                [
                    xb, 
                    kpts.reshape(nb, nt, -1),
                    vel.reshape(nb, nt, -1)
                ],dim=-1)
            
        
        # extract motion embeddings
        motion_emb = self.encode_motion(xs) #[b,d]
        
        # extract text embeddings
        text_emb = self.encode_text(action_labels[:,0]) #[b,d]
        
        
        # compute the contrastive loss
        ## referring to https://openaccess.thecvf.com/content/CVPR2022/supplemental/Guo_Generating_Diverse_and_CVPR_2022_supplemental.pdf
        margin=10
        
        D = torch.cdist(text_emb, motion_emb, p=2)
        D2 = torch.cdist(action_labels.float(), action_labels.float(), p=2)
        y = (D2==0).float()
        
        # Compute the loss for positive pairs: L_pos = y * D^2
        loss_pos = y * (D ** 2)
        
        # Compute the loss for negative pairs:
        # L_neg = (1 - y) * {max(0, margin - D)}^2
        loss_neg = (1 - y) * (F.relu(margin - D) ** 2)
        
        # Combine the losses
        loss = torch.mean(loss_pos + loss_neg)
        
        return loss


    def training_step(self,batch):
        loss = self.forward(batch)
        self.log(
            "contrastiveloss/train", loss, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True
        )

        return loss

    @torch.no_grad()
    def validation_step(self,batch):
        loss = self.forward(batch)
        self.log(
            "contrastiveloss/val", loss, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True
        )

        return loss




    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulersq
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        
        return {"optimizer":optimizer,}



    @torch.no_grad()
    def canonicalized_customized_mc(self,batch):
        
        (
            betas_all, 
            xb_all, 
            action_labels,
         ) = (
                batch["betas"], 
                batch["xb"], 
                batch["action_label"]
         )
        
        
        # canonicalize
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
        xb = xb[:,:-1]
        kpts = kpts[:,:-1]
        nb, nt = xb.shape[:2]
        
        xs = torch.cat(
            [
                xb, 
                kpts.reshape(nb, nt, -1),
                vel.reshape(nb, nt, -1)
            ],dim=-1)
            
        
        return xs
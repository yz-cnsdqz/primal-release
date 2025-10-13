import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def skew_symmetric_matrix(x):
    """
    Generate the skew-symmetric matrix [x]_cross for a batch of 3D vectors.
    Args:
        x: Tensor of shape [b, 3], where b is the batch size.
    Returns:
        skew_matrices: Tensor of shape [b, 3, 3] representing the skew-symmetric matrices.
    """
    # Ensure the input has shape [b, 3]
    assert x.shape[-1] == 3, "Input tensor must have shape [b, 3]"

    # Extract components of the vector
    x1 = x[:, 0]  # Shape [b]
    x2 = x[:, 1]  # Shape [b]
    x3 = x[:, 2]  # Shape [b]

    # Construct the skew-symmetric matrix batch
    zero = torch.zeros_like(x1)  # Shape [b]
    
    skew_matrices = torch.stack([
        torch.stack([zero, -x3, x2], dim=-1),   # Row 1
        torch.stack([x3, zero, -x1], dim=-1),  # Row 2
        torch.stack([-x2, x1, zero], dim=-1)   # Row 3
    ], dim=1)  # Stack rows along dimension 1 to form the matrix

    return skew_matrices  # Shape [b, 3, 3]




def feat_transf(x, beta, gamma):
    return x * (1 + gamma) + beta




class TimeEmbedder(nn.Module):
    """
    Following DIT (facebook), the time embedding is sinusoid + mlp
    """
    def __init__(self, d_model=256, max_len=5000):
        super(TimeEmbedder, self).__init__()
        # frequency embedding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t_idx):
        '''
        t_idx: [b]
        '''
        x_emb = self.pe[t_idx] #[b,d]
        x_emb = self.mlp(x_emb) #[b,d]

        return x_emb




class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        seq_len = x.size(1)
        seq_dim = x.size(2)
        
        x = x + self.pe[:,:seq_len, :seq_dim]
        return self.dropout(x)




class AdaLN0Attn(nn.Module):
    '''
    https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf
    '''
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6, 
        ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, 
            nhead, 
            dropout=dropout,
            bias=True, 
            batch_first=True,)
        # Implementation of Feedforward model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=layer_norm_eps)
        
        self.mlp_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)
        )
        self._init_weight()

    def _init_weight(self,):
        nn.init.constant_(self.mlp_adaLN[-1].weight, 0)
        nn.init.constant_(self.mlp_adaLN[-1].bias, 0)
        

    def forward(
            self, 
            cond, 
            x, 
            attn_mask=None,
    ):
        beta_attn, gamma_attn, alpha_msa, beta_mlp, gamma_mlp, alpha_mlp = self.mlp_adaLN(cond).chunk(6, dim=-1)
        hh = feat_transf(self.norm1(x), beta_attn, gamma_attn)
        x = x + alpha_msa * self.attn(hh,hh,hh,
                                      attn_mask=attn_mask)[0]
        x = x + alpha_mlp * self.mlp(feat_transf(self.norm2(x), beta_mlp, gamma_mlp))
        return x





class AdaLN0MLP(nn.Module):
    '''
    https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf
    '''
    def __init__(self, 
                 d_model: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6, 
        ):
        super().__init__()

        self.mlp0 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        # Implementation of Feedforward model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=layer_norm_eps)
        
        self.mlp_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)
        )
        self._init_weight()

    def _init_weight(self,):
        nn.init.constant_(self.mlp_adaLN[-1].weight, 0)
        nn.init.constant_(self.mlp_adaLN[-1].bias, 0)
        

    def forward(
            self, 
            cond, 
            x, 
    ):
        beta_attn, gamma_attn, alpha_msa, beta_mlp, gamma_mlp, alpha_mlp = self.mlp_adaLN(cond).chunk(6, dim=-1)
        hh = feat_transf(self.norm1(x), beta_attn, gamma_attn)
        x = x + alpha_msa * self.mlp0(hh)
        x = x + alpha_mlp * self.mlp(feat_transf(self.norm2(x), beta_mlp, gamma_mlp))
        return x




class AdaLN0Outlayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, d_model: int, d_out: int):
        super().__init__()

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, d_out)
        self.mlp_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model)
        )

    def _init_weight(self,):
        nn.init.constant_(self.mlp_adaLN[-1].weight, 0)
        nn.init.constant_(self.mlp_adaLN[-1].bias, 0)
        

    def forward(self, cond, x):
        beta, gamma = self.mlp_adaLN(cond).chunk(2, dim=-1)
        x = feat_transf(self.norm(x), beta, gamma)
        x = self.linear(x)
        return x



class TransformerAdaLN0(nn.Module):
    def __init__(
        self,
        in_dim = 128,
        out_dim = 128,
        h_dim = 512,
        n_layer = 4,
        n_head = 8,
        n_time_embeddings=1000,
        dropout = 0.25,
        use_positional_encoding=True,
    ):
        super().__init__()
        self.use_pe = use_positional_encoding

        # time
        self.time_embedding = TimeEmbedder(max_len=n_time_embeddings, 
                                           d_model=h_dim)
        
        # positional encoding
        if use_positional_encoding:
            self.pos_embedding = PositionalEncoder(h_dim, dropout=0)
        

        # transformer
        self.in_fc = nn.Linear(in_dim, h_dim)
        self.tf_encoder = nn.ModuleList(
            [
                AdaLN0Attn(h_dim, n_head, dropout=dropout) 
                for _ in range(n_layer)
            ]
        )

        self.out_fc = AdaLN0Outlayer(h_dim, out_dim)
        


    def forward(
        self,
        timestep,
        sample, #[b,t,d]
        c_emb, #[b,t,d]
        attn_mask=None,
    ):
        
        # 1. time. Note this time is from the diffusion process, not the sequence time!
        t_emb = self.time_embedding(timestep).unsqueeze(1)

        # 1.1 following MDM, we add time embedding to condition embedding
        tc_emb = c_emb + t_emb

        # 2. encoding motion patterns
        hxt = self.in_fc(sample) 

        # 3. apply positional encoding
        if self.use_pe:
            hxt = self.pos_embedding(hxt)

        # 4. fwd
        for layer in self.tf_encoder:
            hxt = layer(tc_emb, 
                        hxt, 
                        attn_mask=attn_mask)

        # 5. pass to the transformer encoder
        output = self.out_fc(tc_emb, hxt)
        return output




class TransformerInContext(nn.Module):
    def __init__(
        self,
        in_dim = 128,
        out_dim = 128,
        h_dim = 512,
        n_layer = 4,
        n_head = 8,
        n_time_embeddings=1000,
        dropout = 0.25,
        separate_condition_token=True,
        use_positional_encoding=True,
        act_fun = 'relu',
    ):
        super().__init__()
        self.separate_condition_token = separate_condition_token
        self.use_pe = use_positional_encoding
        if act_fun == 'relu':
            activation = F.relu
        elif act_fun == 'gelu':
            activation = F.gelu
        elif act_fun == 'silu':
            activation = F.silu
        else:
            raise NotImplementedError

        # time
        self.time_embedding = TimeEmbedder(max_len=n_time_embeddings, 
                                           d_model=h_dim)
        
        # positional encoding
        if use_positional_encoding:
            self.pos_embedding = PositionalEncoder(h_dim, dropout=0)
        

        # transformer
        self.in_fc = nn.Linear(in_dim, h_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=n_head,
                                               batch_first=True, dropout=dropout,
                                               norm_first=True,
                                               activation=activation
                                               )
        self.tf_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer,
                                                enable_nested_tensor=False)
        
        self.out_fc = nn.Linear(h_dim, out_dim)
        


    def forward(
        self,
        timestep,
        sample, #[b,t,d]
        c_emb, #[b,t,d]
        attn_mask=None,
        control_emb=None,
        control_weight=0.0,
    ):        
        # 1. time. Note this time is from the diffusion process, not the sequence time!
        t_emb = self.time_embedding(timestep).unsqueeze(1)
        if control_emb is not None:
            c_emb = c_emb + control_emb*control_weight

        # 2. encoding motion patterns
        hxt = self.in_fc(sample) 

        # 3. cat them along the time dimension
        if self.separate_condition_token:
            hxt = torch.cat([t_emb, c_emb, hxt],dim=1)
        else:
            hxt = torch.cat([t_emb+c_emb, hxt],dim=1)
        

        # 3. apply positional encoding
        if self.use_pe:
            hxt = self.pos_embedding(hxt)
        
        # 4. fwd
        nt = hxt.shape[1]
        if attn_mask is not None:
            attn_mask = attn_mask[:nt, :nt] 
        if self.separate_condition_token:
            hxt = self.tf_encoder(hxt,mask=attn_mask)[:,2:]
        else:
            hxt = self.tf_encoder(hxt,mask=attn_mask)[:,1:]

        
        # 5. pass to the transformer encoder
        output = self.out_fc(hxt)
        return output




#================= the controlnet implementation ====================#

class TransformerInContextControlNet1(nn.Module):
    def __init__(
        self,
        in_dim = 128,
        out_dim = 128,
        h_dim = 512,
        n_layer = 4,
        n_head = 8,
        n_time_embeddings=1000,
        dropout = 0.25,
        separate_condition_token=True,
        use_positional_encoding=True,
        act_fun='relu',
    ):
        super().__init__()
        self.separate_condition_token = separate_condition_token
        self.use_pe = use_positional_encoding

        if act_fun == 'relu':
            activation = F.relu
        elif act_fun == 'gelu':
            activation = F.gelu
        elif act_fun == 'silu':
            activation = F.silu
        else:
            raise NotImplementedError


        # time
        self.time_embedding = TimeEmbedder(max_len=n_time_embeddings, 
                                           d_model=h_dim)
        
        # positional encoding
        if use_positional_encoding:
            self.pos_embedding = PositionalEncoder(h_dim, dropout=0)
        

        # transformer
        self.in_fc = nn.Linear(in_dim, h_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=n_head,
                                               batch_first=True, dropout=dropout,
                                               norm_first=True,
                                               activation=activation)
        self.tf_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer,
                                                enable_nested_tensor=True)
        
        self.out_fc = nn.Linear(h_dim, out_dim)
        
        # controller
        self.ctrl_tf_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h_dim, 
                nhead=n_head,
                batch_first=True, 
                dropout=dropout,
                norm_first=True,
                activation=activation,
            ), 
            num_layers=n_layer,
            enable_nested_tensor=True
        )

        self.ctrl_encoder_out = nn.ModuleList(
            [
                nn.Linear(h_dim, h_dim) 
                for _ in range(n_layer)
            ]
        )


    def init_weights(self):
        #Copy parameters from tf_encoder to ctrl_encoder.
        for encoder_layer, controller_layer in zip(self.tf_encoder.layers, self.ctrl_tf_encoder.layers):
            controller_layer.load_state_dict(encoder_layer.state_dict())

        # Initialize ctrl_encoder_out with 0s
        for layer in self.ctrl_encoder_out:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)
        
        print('-- weights initialized')


    def get_ft_params(self):
        """Get parameters from ctrl_encoder layers for optimization."""
        params = []
        for layer in self.ctrl_tf_encoder.layers:
            params.extend(layer.parameters())
        for layer in self.ctrl_encoder_out:
            params.extend(layer.parameters())

        return params


    def forward_basemodel(
        self,
        timestep,
        sample, #[b,t,d]
        c_emb, #[b,t,d]
        attn_mask=None,
        control_list=None,
        control_weight=1.0,
    ):        
        # 1. time. Note this time is from the diffusion process, not the sequence time!
        t_emb = self.time_embedding(timestep).unsqueeze(1)
        
        # 2. encoding motion patterns
        hxt = self.in_fc(sample) 

        # 3. cat them along the time dimension
        if self.separate_condition_token:
            hxt = torch.cat([t_emb, c_emb, hxt],dim=1)
        else:
            hxt = torch.cat([t_emb+c_emb, hxt],dim=1)
        
        # 3. apply positional encoding
        if self.use_pe:
            hxt = self.pos_embedding(hxt)

        # 4. fwd
        for layer, control in zip(self.tf_encoder.layers, control_list):
            hxt = layer(hxt, src_mask=attn_mask) + control * control_weight

        if self.separate_condition_token:
            hxt = hxt[:,2:]
        else:
            hxt = hxt[:,1:]

        # 5. pass to the transformer encoder
        output = self.out_fc(hxt)
        return output


    def forward_controlnet(
        self,
        timestep,
        sample, #[b,t,d]
        c_emb, #[b,t,d]
        control_emb=None,
        attn_mask=None,
    ):
        # 1. time. Note this time is from the diffusion process, not the sequence time!
        t_emb = self.time_embedding(timestep).unsqueeze(1)
        
        # 2. encoding motion patterns
        hxt = self.pos_embedding(self.in_fc(sample)+control_emb)
        
        # 3. cat them along the time dimension
        if self.separate_condition_token:
            hxt = torch.cat([t_emb, c_emb, hxt],dim=1)
        else:
            hxt = torch.cat([t_emb+c_emb, hxt],dim=1)

        # 4. fwd
        intermediate_outputs = []
        # You can hook into each layer's output here
        for layer, fcout in zip(self.ctrl_tf_encoder.layers, self.ctrl_encoder_out):
            hxt = layer(hxt,src_mask=attn_mask)
            cout = fcout(hxt)
            intermediate_outputs.append(cout)  # Collect

        
        return intermediate_outputs


    def forward(
        self,
        timestep,
        sample, #[b,t,d]
        c_emb, #[b,t,d]
        control_emb=None, # action embedding etc.
        attn_mask=None,
        control_weight=1.0,
    ):

        control_list = self.forward_controlnet(
            timestep,
            sample, #[b,t,d]
            c_emb, #[b,t,d]
            control_emb=control_emb,
            attn_mask=None,
        )

        output = self.forward_basemodel(
            timestep,
            sample, #[b,t,d]
            c_emb, #[b,t,d]
            control_list=control_list,
            attn_mask=None,
            control_weight = control_weight,
        )

        return output






class TransformerInContextControlNet2(nn.Module):
    """
    Our proposed.
    Similar to OMG, each block of the controlnet transformer takes from its previous basemodel.

    """
    def __init__(
        self,
        in_dim = 128,
        out_dim = 128,
        h_dim = 512,
        n_layer = 4,
        n_head = 8,
        n_time_embeddings=1000,
        dropout = 0.25,
        separate_condition_token=True,
        use_positional_encoding=True,
        act_fun = 'relu',
    ):
        super().__init__()
        self.separate_condition_token = separate_condition_token
        self.use_pe = use_positional_encoding
        assert self.separate_condition_token==False, 'separate token is not supported.'
        if act_fun == 'relu':
            activation = F.relu
        elif act_fun == 'gelu':
            activation = F.gelu
        elif act_fun == 'silu':
            activation = F.silu
        else:
            raise NotImplementedError

        # time
        self.time_embedding = TimeEmbedder(max_len=n_time_embeddings, 
                                           d_model=h_dim)
        
        # positional encoding
        if use_positional_encoding:
            self.pos_embedding = PositionalEncoder(h_dim, dropout=0)
        

        # transformer
        self.in_fc = nn.Linear(in_dim, h_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=n_head,
                                               batch_first=True, dropout=dropout,
                                               norm_first=True,
                                               activation=activation,
                                               )
        self.tf_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer,
                                                enable_nested_tensor=False)
        
        self.out_fc = nn.Linear(h_dim, out_dim)
        
        
        self.ctrl_tf_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h_dim,
                nhead=n_head,
                batch_first=True, 
                dropout=dropout,
                norm_first=True,
                activation=activation,
            ),
            num_layers=n_layer,
            enable_nested_tensor=False
        )

        self.ctrl_encoder_out = nn.ModuleList(
            [
                nn.Linear(h_dim, h_dim) 
                for _ in range(n_layer)
            ]
        )


    def init_weights(self):
        #Copy parameters from tf_encoder to ctrl_encoder.
        for encoder_layer, controller_layer in zip(self.tf_encoder.layers, 
                                                   self.ctrl_tf_encoder.layers
                                                   ):
            controller_layer.load_state_dict(encoder_layer.state_dict())

        # Initialize ctrl_encoder_out with 0s
        for layer in self.ctrl_encoder_out:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)
        
        print('-- weights initialized')


    def get_ft_params(self):
        """Get parameters from ctrl_encoder layers for optimization."""
        params = []
        for layer in self.ctrl_tf_encoder.layers:
            params.extend(layer.parameters())
        for layer in self.ctrl_encoder_out:
            params.extend(layer.parameters())
        
        return params


    def forward(
        self,
        timestep,
        sample, #[b,t,d]
        c_emb, #[b,t,d]
        attn_mask=None,
        control_emb=None,
        control_weight=1.0,
    ):        
        # 1. time. Note this time is from the diffusion process, not the sequence time!
        t_emb = self.time_embedding(timestep).unsqueeze(1)
        
        # 2. encoding motion patterns
        hxt = self.in_fc(sample) 

        # 3. cat them along the time dimension
        if self.separate_condition_token:
            hxt = torch.cat([t_emb, c_emb, hxt],dim=1)
        else:
            hxt = torch.cat([t_emb+c_emb, hxt],dim=1)
        
        # 3. apply positional encoding
        if self.use_pe:
            hxt = self.pos_embedding(hxt)

        # 4. fwd
        for layer, layer_c, layer_c_out in zip(
                            self.tf_encoder.layers, 
                            self.ctrl_tf_encoder.layers,
                            self.ctrl_encoder_out,
                            ):
            hxt_o = layer(hxt, src_mask=attn_mask)
            hxt_c = layer_c(hxt + control_emb*control_weight)
            hxt = hxt_o + layer_c_out(hxt_c) * control_weight

        
        if self.separate_condition_token:
            hxt = hxt[:,2:]
        else:
            hxt = hxt[:,1:]

        # 5. pass to the transformer encoder
        output = self.out_fc(hxt)

        return output





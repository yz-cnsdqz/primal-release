# PRIMAL: Physically Reactive and Interactive Motor Model for Avatar Learning


This repo is developed based on pytorch-lightning, hydra, huggingface libs, and others.

[project page](https://yz-cnsdqz.github.io/eigenmotion/PRIMAL/)





## Unreal Engine Demo
This code repo and downloaded checkpoint serve as the backend for the Unreal Engine demo. 
To run the full UE demo, one can follow [this instruction](https://meshcapade.notion.site/runbook-for-primal-unreal-engine-demo?source=copy_link) to download and setup the frontend.


## 0. Installation

### 0.1 Required SMPL-X Models

This project requires **two SMPL-X model files** for different purposes:

1. **Standard SMPL-X Model** (for motion generation):
   - **File**: `models/SMPLX/neutral/SMPLX_neutral.npz`
   - **Source**: [SMPL-X Official Website](https://smpl-x.is.tue.mpg.de)
   - **Purpose**: Core motion generation and body model computations
   - **Required for**: All training and inference operations

2. **Slim SMPL-X Model** (for visualization/rendering):
   - **File**: `models/SMPLX/SMPLX_neutral_array_f32_slim.npz`
   - **Source**: Cannot be provided directly. Needs [this script](https://github.com/Meshcapade/smpl-rs/blob/main/misc_scripts/standardize_smpl.py) to process the downloaded SMPL-X checkpoint.
   - **Purpose**: Optimized for headless rendering with gloss-rs/smpl-rs
   - **Required for**: Video rendering and visualization
   

**Model Registry Structure:**
```
model-registry/
└── models/
    └── SMPLX/
        ├── neutral/
        │   └── SMPLX_neutral.npz                    # Standard model (download required)
        └── SMPLX_neutral_array_f32_slim.npz         # Slim model (for rendering)
```

### 0.2 Environment Configuration

When using `python-dotenv`, set up your data paths by copying and editing the environment template:
```bash
cp .env.example .env
# Edit .env file with your specific paths:
# AMASS_DATA_PATH=/your/path/to/AMASS/data
# MODEL_REGISTRY_PATH=/your/path/to/model-registry
```

### 0.3 Poetry
In case of no poetry, you can first install or update your poetry via 
```
poetry self update
```
See more info at [Poetry's main page](https://python-poetry.org).

If the first time to use, enter the project root directory and type
```
poetry install
```
You might need to specify the intallation path if you want. 


To modify the repo, e.g. install another specific library or update some versions, use
```
poetry add xxx=v1.00
```
Poetry will check version compatibilities automatically, and lock the versions in `poetry.lock`.



### 0.4 env variable setup
To help setup the path, one can specify the paths in `.env.example`, and rename it to `.env`. 
Note that this `.env` is not automatically loaded. It is necessary to use `python-dotenv` to handle this additionally.



### 0.5 pre-trained checkpoints

The pretrained checkpoints are saved in `logs/`. The base models are saved in `logs/motion_diffuser_ar_30fps/runs`, and the personalized models are saved in `logs/motion_diffuser_ar_action/runs`. 




## 1. Inference with Gradio Demos

We leverage [gradio](https://www.gradio.app) to show how our methods work. The gradio demo files are self-explained. One can check more details there.

To run the gradio demo only with the base model, in your terminal type e.g.
```
poetry run python demos/ARDiffusion_gradio.py logs/primal/motion_diffuser_ar_30fps/runs/silu_anchor
```
The results, i.e. the set of (.pkl, .mp4, and .smpl) are saved to `outputs/gradio_ARDiffusion`. One can change the output folder inside of the gradio demo file.



To run the gradio demo of personlized actions, run 
```
poetry run python demos/ARDiffusionAction_gradio.py logs/motion_diffuser_ar_action/runs/ours
```
The results are automatically saved to `outputs/gradio_ARDiffusionAction`.





## 2. Inference with Scripts

To speed up computation and systematical evals, one needs to run scripts rather than the gradio demos. All scripts are saved in `scripts/` and experiments in `experiments/`.
One can either run the .py files individually, or run the .sh files that wrap the py files.


### 2.1 The base model
The base model is pretrained on AMASS, and can produce perpetual motions given an initial state.
The generation uses an autoregressive diffusion approach with overlapping sliding windows, where each iteration:
1. **Canonicalizes** motion to a local coordinate frame
2. **Embeds** the motion seed as conditioning
3. **Denoises** via reverse diffusion to generate a motion primitive
4. **Post-processes** to transform back to world coordinates

You can control the motion through:
- **Classifier-based guidance**: Control movement direction and facing orientation
- **Velocity perturbations**: Trigger specific actions (left_kick, right_kick, run_forward, flip_back, roll_forward)

Following are examples about motion generation. Reactions to perturbations and classifier-based control have the identical gen|render|eval steps.
See `run_principledAction.sh` and `run_principledControl.sh` for more details.


#### 2.1.1 run the `.sh` file (recommended)
The `.sh` file wraps python files with default settings. The format is
```
bash scripts/run_motionrealism.sh [mode] [checkpointfolder]
```
Then the script will run corresponding python scripts with default settings. One needs to run the "gen" mode first. For example,
```
bash scripts/run_motionrealism.sh gen logs/motion_diffuser_ar_30fps/runs/silu_anchor
```
Afterwards, run
```
bash scripts/run_motionrealism.sh render logs/motion_diffuser_ar_30fps/runs/silu_anchor
```
to render the videos, or
```
bash scripts/run_motionrealism.sh eval logs/motion_diffuser_ar_30fps/runs/silu_anchor
```
to perform quantitative evaluation.
Note that the eval results will be forwarded to a txt file besides the generated pkl file.


#### 2.1.2 run individual `.py` files:

For more control over parameters, you can run individual Python files:

Step 1: generate motions. Run

```
poetry run python experiments/gen_Motion.py --expdir logs/motion_diffuser_ar_30fps/runs/silu_anchor --ckptidx 29999 --use_ema --dataset SFU --use_reproj_kpts --use_inertialization --frame_skip 400
```
The results are automatically saved into `outputs/MotionRealism/{checkpointfoldername}`.

Step 2: quantitative evaluation. Run the following script, and the results will be printed in the terminal
```
python experiments/eval_generation.py outputs/MotionRealism/silu_anchor/SFU_ema-True_reproj-True_inertial-True.pkl
```

Step 3: render the videos. The videos will save besides where the pkl file is located. See `primal/rendering/render.py` for its options.
```
python primal/rendering/render.py outputs/MotionRealism/silu_anchor/SFU_ema-True_reproj-True_inertial-True.pkl false
```



### 2.2 The personalized action model
The personalized action model is obtained by adapting the base model to personalized action data captured with Mocapade 3.0.
Given an initial state and an action label, it generates arbitrarily long motions belonging to that action class.

**Control mechanisms:**
- **Action embeddings**: Discrete action labels (learned during training) condition the diffusion process
- **Classifier-free guidance**: Adjustable strength of action control via `guidance_weight_action` parameter
- **Velocity perturbations**: Random noise to diversify action execution
- **Optional trajectory control**: Can combine action generation with movement/facing direction guidance

The action labels depend on your personalized data. Our checkpoint uses `['jump', 'kick', 'punch', 'run', 'walk']`.
You are free to modify this in your own scenarios.


#### 2.2.1 run the `.sh` file (recommended)
The `.sh` file wraps python files with default settings. The format is
```
bash scripts/run_ARDiffusionAction.sh [mode] [checkpointfolder]
```
The script will run corresponding python scripts with default settings. One needs to run the "gen" mode first. For example,
```
bash scripts/run_ARDiffusionAction.sh gen logs/motion_diffuser_ar_action/runs/ours
```

Afterwards, run
```
bash scripts/run_ARDiffusionAction.sh render logs/motion_diffuser_ar_action/runs/ours
```
to render the videos, or
```
bash scripts/run_ARDiffusionAction.sh eval logs/motion_diffuser_ar_action/runs/ours
```
to perform quantitative evaluation.
Note that the eval results will be forwarded to a txt file besides the generated pkl file.


#### 2.2.2 run individual `.py` files

For more control over parameters, you can run individual Python files:

Step 1: generate motions. Run

```
python experiments/gen_Motion_ARDiffusionAction.py --expdir logs/motion_diffuser_ar_action/runs/ours --ckptidx 999 --use_ema --dataset SFU --use_reproj_kpts --use_inertialization --frame_skip 400 --action run
```
The results are automatically saved into `outputs/ARDiffusionAction/{checkpointfoldername}`.

Step 2: quantitative evaluation. Run the following script, and the results will be printed in the terminal
```
python experiments/eval_generation.py outputs/ARDiffusionAction/ours/SFU_ema-True_reproj-True_inertial-True_action-run.pkl
```
Depending on how we saved the data, the evaluation script will compute corresponding metrics.
For example, R-precision is computed to evaluate the action classification accuracy.

Step 3: render the videos. The videos will save besides where the pkl file is located. See `primal/rendering/render.py` for its options.
```
python primal/rendering/render.py outputs/ARDiffusionAction/ours/SFU_ema-True_reproj-True_inertial-True_action-run.pkl false
```




## 3. Training

Training configurations are based on hydra. 


### 3.1 Base model pretraining

One can directly type
```
python scripts/train.py --config-name=train_diffusion task_name=motion_diffuser_ar
```
It will load the default configuations in `primal/configs/train_diffusion.yaml`, and save the checkpoint to `logs/motion_diffuser_ar/runs/<year>-<month>-<date>-<hour>-<minute>-<second>`. This folder is exactly the `<checkpoint_path>` for testing.
Detailed default settings of data, model, etc. are in respective config folders. For example, `- data: amass_smplx` in `primal/configs/train_diffusion.yaml` is corresponding to `primal/configs/data/amass_smplx.yaml`.

To keep the default setting consistent, it is suggested to specify the non-default settings in the command line, e.g. 
```
python scripts/train.py --config-name=train_diffusion task_name=motion_diffuser_ar data.subsets=['ACCAD'] model.cfg.scheduler.num_train_timesteps=50 
```
or the following one with more customized configurations
```
python scripts/train.py task_name=motion_diffuser_ar_30fps data.batch_size=512 data.framerate=30 data.seq_len=16 model._target_=primal.models.motion_diffuser.ARDiffusion  model.cfg.scheduler.num_train_timesteps=50 trainer.max_epochs=30000 model.cfg.use_metric_velocity=true data.subsets=['ACCAD','BMLmovi','BMLrub','CMU','DFaust','EKUT','Eyes_Japan_Dataset','GRAB','HDM05','KIT','MoSh','PosePrior','SFU','SOMA','SSM','TCDHands','TotalCapture','Transitions']
```

Note that the pretraining phase will take days until you can get good motions. Therefore, we suggest to run the pretraining on cluster. 
To monitor the training process, you can use tensorboard by typing in the terminal
```
tensorboard --logdir=[path_of_checkpoints]
```



### 3.2 Model Adaptation

In this work, we consider two avatar-related tasks: **spatial target reaching** and **semantic action generation**.

**Adaptation approaches:**
According to the paper, the three approaches are `finetuning`, `OmniControlNet`, and `ours`, which correspond to:
- `finetune`: Fine-tune all model parameters
- `controlnet1`: OmniControl-style ControlNet architecture
- `controlnet2`: Our proposed ControlNet approach (recommended)

Specify the approach via `+model.cfg.network.controltype=[adaptation_approach]`.

**Important:** Always specify the pretrained base model via e.g. `'+finetune_from_exp=logs/motion_diffuser_ar_30fps/runs/silu_anchor'`. Otherwise, the model will train from scratch.


#### 3.2.1 Spatial target reaching
This adaptation enables motion generation towards a specific 3D target location (or 2D target on the xz-plane).
The model learns to embed spatial targets and guide the diffusion process using classifier-free guidance.

To perform adaptation, run the following command for example:
```
python scripts/train.py --config-name=train_diffusion task_name=motion_diffuser_ar_spatial data.batch_size=128 trainer.max_epochs=1000 model._target_=primal.models.motion_diffuser.ARDiffusionSpatial model.cfg.network.type=transformerInContext '+model.cfg.network.controltype=controlnet2' '+model.cfg.goal_type=2D' '+finetune_from_exp=logs/motion_diffuser_ar_30fps/runs/silu_anchor'
```
The checkpoint will be saved to the `task_name`, e.g. `logs/motion_diffuser_ar_spatial`.
One can change the settings like learning rate, batch size, goal_type, etc. accordingly.


#### 3.2.2 semantic action generation based on personalized data.
The base model can be quickly adapted to some user captured motion sequences.
In the following, we share what we do to create a personalized motion model.

##### Step 1: Capture some videos with cellphone.
We use a cellphone to capture some videos of specific actions of martial arts, i.e. ['jump', 'kick', 'punch', 'run', 'walk']. 
You can capture a very long video containing all actions, or capture them individually.
Of course, the action labels are dependent on the scenario.

##### Step 2: Estimate body motions with [Mocapade3.0.](https://me.meshcapade.com/from-videos). 
Trim the videos to 5-7 seconds so that the server can process them. Upload the videos, and then download the `.smpl` files.

##### Step 3: Dataset Structure and Annotation.
**Dataset Directory Structure:**
Create your dataset directory with all `.smpl` files in a single folder:
```
your_dataset_folder/
├── jump_001.smpl
├── jump_002.smpl  
├── kick_001.smpl
├── kick_002.smpl
├── punch_001.smpl
├── run_001.smpl
├── walk_001.smpl
└── ...
```

**File Naming Convention:**
According to the action labels, name each individual `.smpl` file to `[action]_[idx].smpl`, such as `jump_003.smpl` and `kick_004.smpl`. **This file formatting is necessary, since the action labels are directly retrieved from the filenames using `os.path.basename(file).split('.')[0].split('_')[0]`.**

**Configuration:**
Set the path to your dataset folder in your environment variables or directly in the config:
- Environment variable: `CUSTOMIZED_ACTION_PATH=/path/to/your_dataset_folder`  
- Or edit `primal/configs/data/customized_action_mc.yaml` and set `path: /path/to/your_dataset_folder`

**Data Processing:**
The system automatically:
- Extracts action labels from filenames (e.g., "jump" from "jump_001.smpl")
- Segments each `.smpl` file into fixed-length sequences (default: 16 frames)


##### Step 4: Perform adaptation
Run the following command for example. Definitely, their settings can be changed.

```
python primal/scripts/train.py task_name=motion_diffuser_ar_action data=customized_action_mc data.path={your_own_dataset} data.batch_size=16 model._target_=primal.models.motion_diffuser.ARDiffusionAction  trainer.max_epochs=1000  model.cfg.network.type=transformerInContext model.cfg.network.act_fun=silu '+model.cfg.network.controltype=controlnet2' 
```
The checkpoint will be saved to `logs/motion_diffuser_ar_action`.


##### Step 5: run gradio demo for testing.
See Section 1 for details of how to run the gradio demo. Definitely, you need to update the action labels if necessary.
Note that the action labels are sorted based on the alphabetical order.





# License and Citation
See [license file](LICENSE) for more details. Please cite the following work if it is useful for you. Many thanks.
```
 @article{zhang2025primal,
      title={PRIMAL: Physically Reactive and Interactive Motor Model for Avatar Learning},
      author={Zhang, Yan and Feng, Yao and Cseke, Alp{\'a}r and Saini, Nitin and Bajandas, Nathan and Heron, Nicolas and Black, Michael J},
      journal={ICCV},
      year={2025}
  }
```


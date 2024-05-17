# mmDiff
This is the code for the paper "mmDiff: Context and Consistency Awareness for mmWave Human Pose Estimation via Multi-Conditional Diffusion"



## Environment

The code is developed and tested under the following environment:

-   Python 3.8.2
-   PyTorch 1.8.1
-   CUDA 11.6

You can create the environment via:

```bash
conda env create -f environment.yml
```
For manual setup, we build our environment following [P4transformer](https://github.com/hehefan/P4Transformer).


## Phase one
### Phase one Dataset
To download dataset, please refer to [mmBody](https://github.com/Chen3110/mmBody) and [mm-Fi](https://github.com/ybhbingo/MMFi_dataset).

### Phase one Models
Phase one models includes implementation of radar point clouds (PC) encoder and the Global Radar Context (GRC) implementation. For `mmBody` and `mm-Fi`, different models are adopted as to handle PCs collected by different radars.

The fundamental model design can be found in `phase1/mmBody/model/P4TransformerSelf.py` for mmBody and `phase1/mmfi/mmwave_point_transformer.py` for mm-Fi.


### Phase one Training 
To train phase one using mmBody dataset, please run:
```bash
### Train P4Transformer from scratch ###
python phase1/mmBody/radar_p4trans_train.py
```

To train and test phase one using mmFi dataset, please go to mmwave_benchmark.ipynb for training .

### Phase two Dataset preparation 
To facilitate phase two training, pretrained model in phase one are utilized to extract GRC features and coarse human poses $\Tilde{H}$, which are saved to .npz files.


## Phase two

### Phase two Dataset
Add discription about extract from phase one.


We provide the pretrained mmBody dataset as [mmBody.zip]. To run the code, please download all the `mmBody.zip` and extract the .npy files to the `phase2/mmBody/` folder. Meanwhile, to follow the setting of the Human 3.6m dataset of human pose estimation, please download the `pose_format.zip` and extract it to the `phase2/pose_format\` folder. 



### Phase two Models
The fundamental model design can be found in `phase2/models/mmDiff.py`, which includes designs for Local Radar Context (LRC), Temporal Motion Consistency (TMC) and Structural Limb-length Consistency (SLC).
 
We provide the pretrained mmDiff parameter [here]. Before running the code, please download the `checkpoints.zip` and extract all the .pth files to the `phase2/checkpoints/` folder. Before running the code, please specify the checkpoint path in the runner.sh shell file as: 
```bash
--model_diff_path checkpoints/[name].pth \
```


### Evaluating pre-trained models

We provide the pre-trained diffusion model [ckpt_71.pth]. To evaluate it, put it into the `phase2/checkpoint` directory and run:

```bash
### inference with pretrain ###
CUDA_VISIBLE_DEVICES=0 python main_mmDiff.py \
--config mmDiff_config.yml --batch_size 512 \
--model_diff_path checkpoints/ckpt_71.pth \ # add your pretrained model
--doc test --exp exp --ni \
```

### Training models from scratch
To train a model from scratch, run

```bash
### train ###
CUDA_VISIBLE_DEVICES=0 python main_mmDiff.py --train \
--config mmDiff_config.yml --batch_size 512 \
--doc test --exp exp --ni \
```

# 2023-AI-Samsung-Challenge-Image-Quality-Assessment

![image](https://hackmd.io/_uploads/B1mbT0BQC.png)

![image](https://hackmd.io/_uploads/SJCb60BQC.png)

## Datasets

All the models in this project were evaluated on the following datasets:

- [DACON](https://dacon.io/competitions/official/236134/data) (DACON Site)

## Description

The source code for this solution is based on these two repositories:

- [M2 Transformer](https://github.com/aimagelab/meshed-memory-transformer): for image captioning task (task 1)
- [MANIQA](https://github.com/IIGROUP/MANIQA): for image quality assessment task (task 2)


## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/Leejucheon96/2022-AI-Samsung-Challenage.git


# [OPTIONAL] create conda environment
conda create -n 2023samsung python=3.8
conda activate 2023samsung
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -r requirements.txt
```

However, we recommend to use the docker image we provided [here](https://drive.google.com/file/d/1DD8DMOIm-SiVcuparVPb5Hy_3QD3dKPW/view?usp=drive_link). It is already installed all libraries necessary for the training/reproduction of the results. Just download the `quiil_samsung2023_solution.tar` file and perform below command to import the docker image (docker is required to install before doing this):

```
docker load < quiil_samsung2023_solution.tar
```

After that, activate the container:
```
docker run -d --shm-size 8G --gpus '"device=all"' -it --name samsung_quill_container --mount type=bind,source="<absolute folder path of “data” folder>",target=/samsung2023_iqa_solution/data quiil_samsung2023_solution
```

For `data` path, please see how to prepare in the upcomming instructions below.

## How to run

### Data preparation

Step 1: Download the data from the challenge website, including training and testing samples, as below structure:

```
data
|__train
|____ *.jpg
|__test
|____ *.jpg
|__train.csv
|__test.csv
|__train_mos.csv
|__valid_mos.csv
```
Step 2: Then perform the feature extraction for all train and test images (will be used for training image captioning). Do this by the below command:

```
CUDA_VISIBLE_DEVICE=0 python custom_extract_features.py
```

After that, we obtain two hdf5 files: `train.hdf5` and `test.hdf5` for training and testing samples, respectively.

### External code preparation

* Download `scene_graph_benchmark` [here](https://drive.google.com/file/d/16kjIOgBg9cwpF8ay9-g1tdpBOxYleBTR/view?usp=sharing) and put inside the `MANIQA` folder (used for feature extraction purpose)
* Download pre-trained checkpoint [here](https://drive.google.com/file/d/16kjIOgBg9cwpF8ay9-g1tdpBOxYleBTR/view?usp=sharing), and put inside `grid-feats-vqa` folder.

### Training phase for image captioning (task 1)

To train the image captioning task, use the below command:

1. Go to the RSTNet folder:

```
cd RSTNet
```

2. Train the image captioning model:

```
CUDA_VISIBLE_DEVICE=0 python train_transformer.py --exp_name rstnet --batch_size 50 --m 40 --head 8 --features_path train.hdf5
```

3. The checkpoints will be saved at `saved_transformer_models_m2`

### Training phase for image quality assessment (task 2)

To train the image quality assessment task, use the below command:

1. Go to the MANIQA folder:

```
cd MANIQA
```

2. Train the image quality assessment model:

```
CUDA_VISIBLE_DEVICE=0 python train_maniqa_new_vit_Joint2.py
```

3. Checkpoint will be saved at: `MANIQA_new_vit_or_joint2` folder.

### End-to-end inference phase

For reproduction purpose, we provide the best checkpoints for each task:

1. Best checkpoint for task 1 (epoch_ic.pt): [here]()
2. Best checkpoint for task 2 (epoch_iqa.pt): [here]()

Download them at put inside `checkpoints` folder.

Then, run the below command:

```
CUDA_VISIBLE_DEVICES=0 python end_to_end_inference.py
```

Results are stored in `Full_submission.csv`, which is ready for submission. The results helped us to secure the final third ranking (fourth ranking on the final leaderboard)

![image](https://hackmd.io/_uploads/HySEc0rX0.png)

First 20 rows of final submission:

```
img_name,mos,comments
j00zs3u6dr,6.3342686,i love the colors and the clarity of this image
ytv70so3zb,6.0468326,i m not sure what this is but i think it would have been better with a little more contrast
ia9890oozp,6.021466,i like the use of selective desaturation in this photo
xsj81ypx4a,5.4849973,i think this would have worked better in bw with a little more contrast
f23994ghlh,5.716955,i like the use of bw in this photo it looks like a <unk>
olzx3kqzij,6.2476077,i like the use of black and white in this image
pzhsjj4uxo,6.6428676,i like the contrast of the clouds and the <unk>
2mb38w9n83,6.1964545,i like the idea but the image is a bit too dark
cz4tby7pfx,5.6829925,a reasonable use of hdr but not a very engaging photo in my opinion
8099bltbox,5.142827,i like the idea but the image is a bit too dark
4fxgmbki70,5.058069,i m not sure what i m looking at but i m not sure what it is but i think it is a
398sev5zra,5.54492,i like the idea but the image is a little too dark
397dr2njq3,4.183466,i like the idea but the image is a bit too dark
8l24x7ueri,4.505504,i like the use of color in this one
0payszbeft,5.3249483,i really like the texture and the <unk>
5706ozsu0u,4.810112,i like the texture of this image and the <unk>
tvpgpvwu9y,4.365091,i like the idea of this image but the image is a little too dark for my taste i think it would have
0dwf6bs8mo,4.719687,nice use of color and light
lzqw1192vq,5.5744343,i like the colors and the idea of this photo
1usse6jnw8,5.216581,i like the idea but the image is a bit too much for my taste it seems a bit flat
```

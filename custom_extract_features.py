import torch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import sys
sys.path.append("./grid-feats-vqa")
from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)
import glob
import os
import tqdm
import torch.nn as nn
import h5py
import json
import numpy as np
from PIL import Image
import cv2
from inference_config import iqa_config
"""
Create configs and perform basic setups.
"""
config_file = './grid-feats-vqa/configs/X-152-grid.yaml'
data_path = {
    'train': iqa_config.train_dis_path,
    'test': iqa_config.test_dis_path,
}

cfg = get_cfg()
add_attribute_config(cfg)
cfg.merge_from_file(config_file)
# force the final residual block to have dilations 1
cfg.MODEL.RESNETS.RES5_DILATION = 1
cfg.MODEL.WEIGHTS = './grid-feats-vqa/X-152.pth'
cfg.freeze()

model = build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    cfg.MODEL.WEIGHTS, resume=True
)

model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)['model'])

class DataProcessor(nn.Module):
    def __init__(self):
        super(DataProcessor, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.pool(x)
        x = torch.squeeze(x)    # [1, d, h, w] => [d, h, w]
        x = x.permute(1, 2, 0)  # [d, h, w] => [h, w, d]
        return x.view(-1, x.size(-1))   # [h*w, d]

processor = DataProcessor()
for split in data_path:
    annotation_path = './' + split + '.json'
    images_data = json.load(open(annotation_path))['images']
    saved_features_path = './' + split + '.hdf5'
    img_paths = glob.glob(data_path[split] + '*')
    print("Extracting", split, "...")
    with h5py.File(saved_features_path, 'w') as f:
        for sample in tqdm.tqdm(images_data):
            img_path = data_path[split] + '/' + os.path.basename(sample['file_name'])
            image = Image.open(img_path)
            width, height = image.size
            image = np.array(image)
            image = cv2.resize(image, (512, 512))
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)
            image = image.transpose(2,0,1)
            images = model.preprocess_image([{"image": torch.from_numpy(image)}])
            features = model.backbone(images.tensor)
            outputs = model.roi_heads.get_conv5_features(features)
            img_feat = processor(outputs)
            f.create_dataset('%d_grids' % sample['id'], data=img_feat.detach().cpu().numpy())
        f.close()

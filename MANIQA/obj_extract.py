import sys;

sys.path.append('./scene_graph_benchmark')
from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir

import os
import glob
import cv2
import torch
from PIL import Image
import numpy as np
import tqdm
import json
import h5py
import torch.nn as nn
from tqdm import tqdm
import os

import json
import numpy as np
from bounding_box import bounding_box as bb
import random
def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

colors = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver']
def extract_features(img_path, transforms, model):
    model.eval()
    try:
        image = cv2.imread(img_path)
        img_input = cv2Img_to_Image(image)
    except:
        image = Image.open(img_path)
        img_input = np.array(image)
        if img_input.shape[-1] < 3:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
        img_input = cv2Img_to_Image(img_input)

    img_fit_to_get_clip = img_input
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(cfg.MODEL.DEVICE)
    raw_height, raw_width = img_input.shape[-2:]

    with torch.no_grad():
        prediction = model(img_input.type(torch.FloatTensor))[0].to('cpu')

    prediction = prediction.resize((raw_width, raw_height))
    det_dict = {key: prediction.get_field(key) for key in prediction.fields()}
    box_features = det_dict['box_features']
    boxes_all = det_dict['boxes_all']

    # viz_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # for coords, score in zip(boxes_all, det_dict["scores_all"]):
    #     max_score_index = torch.max(score, dim=-1).indices
    #     viz_box = coords[max_score_index]
    #     try:
    #         bb.add(viz_img, int(viz_box[0]), int(viz_box[1]), int(viz_box[2]), int(viz_box[3]), "obj", colors[random.randint(0, len(colors)-1)])
    #     except:
    #         pass

    return box_features, None


def main():
    # Setting configuration
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    # Configuring VinVl
    cfg.merge_from_file('./scene_graph_benchmark/sgg_configs/vgattr/vinvl_x152c4.yaml')
    argument_list = [
        'MODEL.WEIGHT', './vinvl_vg_x152c4.pth',
        'MODEL.ROI_HEADS.NMS_FILTER', 1,
        'MODEL.ROI_HEADS.SCORE_THRESH', 0.2,
        'TEST.IGNORE_BOX_REGRESSION', False,
        'MODEL.ATTRIBUTE_ON', True,
        'MODEL.DEVICE', 'cuda:0',
        'TEST.OUTPUT_FEATURE', True,
    ]

    cfg.merge_from_list(argument_list)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR

    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)

    transforms = build_transforms(cfg, is_train=False)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    root_data_dir = '/home/compu/LJC/samsung_LK99/dataset/'
    train_img_names = [os.path.basename(item['file_name']) for item in json.load(open(root_data_dir + 'train.json', 'r'))["images"]]
    valid_img_names = [os.path.basename(item['file_name']) for item in
                       json.load(open(root_data_dir + 'valid.json', 'r'))["images"]]
    test_img_names = [os.path.basename(item['file_name']) for item in
                       json.load(open(root_data_dir + 'test.json', 'r'))["images"]]

    filename2id = {
        'train': {os.path.basename(item['file_name']):item['id'] for item in json.load(open(root_data_dir + 'train.json', 'r'))["images"]},
        'valid': {os.path.basename(item['file_name']): item['id'] for item in
                  json.load(open(root_data_dir + 'valid.json', 'r'))["images"]},
        'test': {os.path.basename(item['file_name']): item['id'] for item in
                  json.load(open(root_data_dir + 'test.json', 'r'))["images"]}
    }

    max_objs = 100
    visualize_dir = root_data_dir + '/objs_visualize/'
    if not os.path.exists(visualize_dir):
        os.mkdir(visualize_dir)
    for split, file_names in zip(['valid', 'test'], [valid_img_names, test_img_names]):
        print(split)
        if not os.path.exists(visualize_dir + '/' + split):
            os.mkdir(visualize_dir + '/' + split)
        if split in ['train', 'valid']: split_folder = 'train'
        else: split_folder = 'test'
        data_path = root_data_dir + '/' + split_folder + '/'
        saved_features_path = root_data_dir + '/{0}_objs.hdf5'.format(split)
        with h5py.File(saved_features_path, 'w') as f:
            for img_name in tqdm(file_names):
                feats, visualized_img = extract_features(data_path + '/' + img_name, transforms, model)
                N_objs = feats.shape[0]
                feats = torch.cat([feats, torch.zeros(max_objs - N_objs, 2048)], dim=0)
                f.create_dataset('%d_objs' % filename2id[split][img_name], data=feats.detach().cpu().numpy())
            f.close()

if __name__ == '__main__':
    main()
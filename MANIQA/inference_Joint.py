import os
import torch
import numpy as np
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize, five_point_crop, sort_file
# from data.pipal22_test import PIPAL22
from tqdm import tqdm
from models.maniqa import MANIQA, MANIQA_new_vit_or_ceo, MANIQA_new_vit_ceo_obj
import sys
# GLOBAL config
sys.path.append('/home/compu/LJC/')
import inference_config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_epoch(config, net, test_loader):
    with torch.no_grad():
        net.eval()
        name_list = []
        pred_list = []
        with open(config.valid_path + '/output.txt', 'w') as f:
            for data in tqdm(test_loader):
                pred = 0
                for i in range(config.num_avg_val):
                    x_d = data[0]['d_img_org'].cuda()
                    x_g = data[-1].cuda()
                    pred += net(x_d, x_g)[0]

                pred /= config.num_avg_val
                d_name = data[0]['d_name']
                pred = pred.cpu().numpy()
                name_list.extend(d_name)
                pred_list.extend(pred)
            for i in range(len(name_list)):
                f.write(name_list[i] + ',' + str(pred_list[i]) + '\n')
            print(len(name_list))
        f.close()

def output_submission_dacon_challenge(config):
    template_submission = open(config['path_template'][0]).read().split('\n')[1:]
    template_submission = [row.split(',') for row in template_submission]
    new_submission = open(config['path_output_submission'][0], 'w')

    maniqa_results = open('./output.txt', 'r').read().split('\n')[:-1]
    maniqa_results = [row.split(',') for row in maniqa_results]
    dict_maniqua = dict()

    for row in maniqa_results:
        dict_maniqua[row[0].split('.')[0]] = row[1]

    new_submission.write('img_name,mos,comments\n')
    for row in template_submission[:-1]:
        if row[0] not in dict_maniqua:
            new_submission.write(row[0]+','+row[1]+','+'Nice comments'+'\n')
        else:
            new_submission.write(row[0]+','+dict_maniqua[row[0]]+','+'Nice comments'+'\n')
    new_submission.close()
    print("Inference of image quality prediction is done!")
    return

if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "db_name": "samsung",
        "test_dis_path": "../samsung_LK99/dataset/test/",
        "dis_test_path": "../samsung_LK99/dataset/test.csv",
        
        # optimization
        "batch_size": 1,
        "num_avg_val": 1,
        "crop_size": 384,

        # device
        "num_workers": 8,

        # model
        # model
        "patch_size": 16,
        "img_size": 384,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,

        # load & save checkpoint
        "valid": "/data2/samsung_IC/MANIQA",
        "valid_path": "/data2/samsung_IC/MANIQA",
        "model_path": "/data2/samsung_IC/MANIQA_new_vit_or_joint4/samsung/samsung/epoch44.pt",

        # for challenge submission
        'path_template': inference_config.path_template,
        'path_output_submission': inference_config.path_output_mos_submission,
    })

    if not os.path.exists(config.valid):
        os.mkdir(config.valid)

    if not os.path.exists(config.valid_path):
        os.mkdir(config.valid_path)

    from data_name.SAMSUNG import samsung
    dis_test_path = config["dis_test_path"]
    Dataset = samsung

    # data load
    # test_dataset = Dataset.samsung_test(
    #     csv_file=dis_test_path,
    #     root=config["test_dis_path"],
    #     transform=transforms.Compose([
    #         Normalize(0.5, 0.5), ToTensor()]),
    # )

    test_dataset = Dataset.samsungwithgrid_test(
        csv_file=dis_test_path,
        json_path='../samsung_LK99/dataset/test.json',
        root=config["test_dis_path"],
        transform=transforms.Compose([
            Normalize(0.5, 0.5), ToTensor()]),
        detections_path='../samsung_LK99/dataset/test.hdf5',
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )

    print("Testing at checkpoint", config["model_path"])

    net = MANIQA_new_vit_ceo_obj(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale).cuda()
    
    net.load_state_dict(torch.load(config.model_path))
    losses, scores = [], []
    eval_epoch(config, net, test_loader)
    sort_file(config.valid_path + '/output.txt')
    output_submission_dacon_challenge(config)
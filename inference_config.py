import sys
sys.path.append("./MANIQA")
from MANIQA.config import Config

inferece_path_config = Config({
    "path_template": './template.csv',
    "path_saved_final_submission": './full_submission.csv',
    'path_saved_tmp_caption': './tmp_predicted_captions.json',
})

ic_config = Config({
    "features_path": '../data_samsung/test.hdf5',
    "checkpoint_path": '/data2/samsung_IC/RSTNet/saved_transformer_models_m2/rl25.pth',
    'path_meta_test_data': './test.json',
    'MAX_LEN': 23
})

iqa_config = Config({
    "db_name": "samsung",
    "train_dis_path": "../data_samsung/train/",
    "dis_train_path": "../data_samsung/train.csv",
    "test_dis_path": "../data_samsung/test/",
    "dis_test_path": "../data_samsung/test.csv",
    "batch_size": 1,
    "num_avg_val": 1,
    "crop_size": 384,
    "num_workers": 8,
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
    "valid_path": "./",
    "model_path": "./checkpoints/epoch_iqa.pt",
})

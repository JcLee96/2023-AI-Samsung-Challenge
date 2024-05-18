# GLOBAL config
# import sys

import torch
from tqdm import tqdm
import pickle
import numpy as np
import itertools
import json
import os
import h5py
import csv
from inference_config import inferece_path_config, ic_config, iqa_config
from torchvision import transforms

# Libraries for image captioning model
import sys
sys.path.append('./RSTNet')
from RSTNet.models.m2_transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from RSTNet.data import TextField

# Libraries for IQA
sys.path.append('./MANIQA')
from MANIQA.models.maniqa import MANIQA_new_vit_or_ceo
from MANIQA.data_name.SAMSUNG import samsung
from MANIQA.utils.inference_process import ToTensor, Normalize

def eval_iqa_model(config, net, test_loader):
    # Inference IQA model
    print("Inference for image quality assessment model")
    dict_result_iqas = dict()
    with torch.no_grad():
        net.eval()
        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                pred += net(x_d.unsqueeze(0))[0]
            pred /= config.num_avg_val
            d_name = data['d_name']
            pred = pred.cpu().numpy()
            dict_result_iqas[d_name.split('.')[0]] = pred
        print(len(dict_result_iqas))
    
    return dict_result_iqas

def eval_ic_model(ic_config, ic_net, image_ids, text_field, inferece_path_config, features_test, dict_imgname_id):
    # Inference captioning model
    print("Inference for image captioning model")
    results = []
    for image_id in tqdm(image_ids):
        image = features_test['%d_grids' % image_id][()]
        torch_image = torch.from_numpy(np.array([image])).cuda()
        with torch.no_grad():
            out, _ = ic_net.beam_search(torch_image, ic_config.MAX_LEN, text_field.vocab.stoi['</s>'], 3, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        gen_i = ' '.join([k for k, _ in itertools.groupby(caps_gen[0])])
        gen_i = gen_i.strip().replace('_',' ')
        results.append({"id": dict_imgname_id[image_id], "captions": gen_i})

    # Save results
    json.dump(results, open(inferece_path_config.path_saved_tmp_caption, 'w'), indent=4)
    captions = json.load(open(inferece_path_config.path_saved_tmp_caption, 'r'))
    captions = {caption['id'].split('.')[0]:caption['captions'] for caption in captions}
    return captions

def main():
    print("Preparing captioning model ...")
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
    text_field.vocab = pickle.load(open('./RSTNet/vocab.pkl', 'rb'))
    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                        attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 90, 3, text_field.vocab.stoi['<pad>'])
    ic_net = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).cuda()
    data = torch.load(ic_config.checkpoint_path)
    ic_net.load_state_dict(data["state_dict"])
    print("Preparing captioning model sucessfully!")

    print("Preparing IQA model ...")
    iqa_net = MANIQA_new_vit_or_ceo(embed_dim=iqa_config.embed_dim, num_outputs=iqa_config.num_outputs, dim_mlp=iqa_config.dim_mlp,
            patch_size=iqa_config.patch_size, img_size=iqa_config.img_size, window_size=iqa_config.window_size,
            depths=iqa_config.depths, num_heads=iqa_config.num_heads, num_tab=iqa_config.num_tab, scale=iqa_config.scale).cuda()
    iqa_net.load_state_dict(torch.load(iqa_config.model_path))
    print("Preparing IQA model sucessfully!")

    # Load features
    print("Preparing grid features for image captioning")
    features_test = h5py.File(ic_config.features_path, 'r')
    image_ids = [i['id'] for i in json.load(open(ic_config.path_meta_test_data))['images']]
    dict_imgname_id = {i['id']: os.path.basename(i['file_name']) for i in json.load(open(ic_config.path_meta_test_data))['images']}

    print("Preparing data loader for IQA")
    # Inference captioning model
    test_dataset = samsung.samsung_test(
        csv_file=iqa_config.dis_test_path,
        root=iqa_config.test_dis_path,
        transform=transforms.Compose([
            Normalize(0.5, 0.5), ToTensor()]),
    )

    imgnames = []
    # Load the test.csv to ensure the order
    with open(iqa_config.dis_test_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(str(row["img_name"])) < 10:
                imgname = ''.join(['0'] * (10 - len(str(row["img_name"])))) + str(row["img_name"])
                imgnames.append(imgname)
            else:
                imgnames.append(str(row["img_name"]))

    # Inference captioning
    if not os.path.isfile(inferece_path_config.path_saved_tmp_caption):
        captions = eval_ic_model(ic_config, ic_net, image_ids, text_field, inferece_path_config, features_test, dict_imgname_id)
    else:
        captions = json.load(open(inferece_path_config.path_saved_tmp_caption))
        captions = {item['id'].split('.')[0]:item['captions'] for item in captions}
        
    # Inference vqa
    result_iqas = eval_iqa_model(iqa_config, iqa_net, test_dataset)

    f = open(inferece_path_config.path_saved_final_submission, 'w')
    f.write('img_name,mos,comments\n')

    for imgname in imgnames:
        f.write(imgname + ',' + str(result_iqas[imgname][0]) + ',' + captions[imgname] + '\n')

    print("Done inference!")

if __name__ == '__main__':
    main()
from data import TextField
import h5py
from models.m2_transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import pickle
import numpy as np
import itertools
import json
import os

# GLOBAL config
import sys
sys.path.append('/home/compu/LJC/')
import inference_config

# Pre-defined paths
path_mos_csv_file = inference_config.path_output_mos_submission
path_saved_final_submission = inference_config.path_output_final_submission
path_saved_tmp_caption = '../dataset/submission/tmp_predicted_captions.json'
features_path = '../dataset/test.hdf5'
checkpoint_path = "./saved_transformer_models/rstnet_best.pth"
language_model_path = './saved_language_models/bert_language_best_test.pth'
MAX_LEN = 23

# Pipeline for text & define model
print("Preparing model...")
text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
 # Model and dataloaders
encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                    attention_module_kwargs={'m': 40})
decoder = MeshedDecoder(len(text_field.vocab), 90, 3, text_field.vocab.stoi['<pad>'])
model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).cuda()
data = torch.load(checkpoint_path)
model.load_state_dict(data["state_dict"])

# Load features
print("Preparing features...")
features_test = h5py.File(features_path, 'r')
image_ids = [i['id'] for i in json.load(open('../dataset/test.json'))['images']]
dict_imgname_id = {i['id']: os.path.basename(i['file_name']) for i in json.load(open('../dataset/test.json'))['images']}

if not inference_config.only_update_mos:
    # Inference
    print("Inference...")
    results = []
    for image_id in tqdm(image_ids):
        image = features_test['%d_grids' % image_id][()]
        torch_image = torch.from_numpy(np.array([image])).cuda()
        with torch.no_grad():
            out, _ = model.beam_search(torch_image, MAX_LEN, text_field.vocab.stoi['</s>'], 3, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        gen_i = ' '.join([k for k, _ in itertools.groupby(caps_gen[0])])
        gen_i = gen_i.strip().replace('_',' ')
        results.append({"id": dict_imgname_id[image_id], "captions": gen_i})

    # Save results
    json.dump(results, open(path_saved_tmp_caption, 'w'), indent=4)
    captions = json.load(open(path_saved_tmp_caption, 'r'))
    captions = {caption['id'].split('.')[0]:caption['captions'] for caption in captions}

else:
    pre_submission = open(inference_config.path_captions_from_submission_csv, 'r').read().split('\n')
    pre_submission = [item.split(',') for item in pre_submission]
    captions = {item[0]:item[-1] for item in pre_submission}

# Make final submission
template_submission = open(path_mos_csv_file[0], 'r').read().split('\n')
template_submission = [item.split(',') for item in template_submission]

f = open(path_saved_final_submission[0], 'w')
f.write('img_name,mos,comments\n')
for row in template_submission[1:-1]:
    if row[0] == '0213074526': # specal case
        f.write(row[0] + ',' + row[1] + ',' + captions['213074526'] + '\n')
    else:
        f.write(row[0] + ',' + row[1] + ',' + captions[row[0]] + '\n')

print("Done inference! Let's submit it!")
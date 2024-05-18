import pickle
import json
import numpy as np
from data import TextField
import tqdm
text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True, nopoints=False)
text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

# text_field.vocab.stoi['breatiful'] = 7288
# text_field.vocab.itos[0] = 'breatiful'

stoi_dict_vocab = {}
itos_dict_vocab = {}
word_idx = 1
for key in text_field.vocab.stoi:
    if key not in ['<unk>', '<pad>', '<bos>', '<eos>']:
        stoi_dict_vocab[key] = word_idx
        itos_dict_vocab[word_idx] = key
        word_idx += 1

stoi_dict_vocab['UNK'] = word_idx
itos_dict_vocab[word_idx] = 'UNK'

f = open('../dataset/vocabulary.txt', 'w')
for word in stoi_dict_vocab.keys():
    f.write(word + '\n')
f.close()

input_seq = {}
target_seq = {}

train_data = json.load(open("../dataset/train.json"))
valid_data = json.load(open("../dataset/valid.json"))

all_captions = []
for sample in train_data["annotations"]:
    all_captions.append(len(sample['caption'].split()))

for sample in valid_data["annotations"]:
    all_captions.append(len(sample['caption'].split()))

max_len = 20
all_captions = []
for sample in tqdm.tqdm(train_data['images']):
    input_list_captions = []
    target_list_captions = []
    gts_captions = []
    image_id = sample['id']
    for annotation in train_data['annotations']:
        if image_id == annotation['image_id']:
            input_list_id_words = [0]
            target_list_id_words = []
            list_words = annotation['caption'].split()
            for word in list_words[:max_len-1]:
                try:
                    input_list_id_words.append(stoi_dict_vocab[word])
                    target_list_id_words.append(stoi_dict_vocab[word])
                except:
                    input_list_id_words.append(0)
                    target_list_id_words.append(0)

            tmp_target_list_id_words = target_list_id_words.copy()
            if tmp_target_list_id_words[-1] != 0:
                tmp_target_list_id_words += [0]
            gts_captions.append(tmp_target_list_id_words)

            input_list_id_words += [0] * (max_len - len(list_words) - 1)
            target_list_id_words += [0]
            target_list_id_words += [-1] * (max_len - len(list_words) - 1)
            
            if len(input_list_id_words) != 20 or len(target_list_id_words) != 20:
                import pdb; pdb.set_trace()
            
            input_list_captions.append(np.array(input_list_id_words))
            target_list_captions.append(np.array(target_list_id_words))
    
    all_captions.append(gts_captions)
    input_seq[image_id] = np.array(input_list_captions)
    target_seq[image_id] = np.array(target_list_captions)

f = open('../dataset/stoi_dict_vocab.json', 'w')
json.dump(stoi_dict_vocab, f)
f.close()

f = open('../dataset/itos_dict_vocab.json', 'w')
json.dump(itos_dict_vocab, f)
f.close()

with open('../dataset/dacoisamsung_input_seq.pkl', 'wb') as handle:
    pickle.dump(input_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../dataset/dacoisamsung_target_seq.pkl', 'wb') as handle:
    pickle.dump(target_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../dataset/dacoisamsung_gts.pkl', 'wb') as handle:
    pickle.dump(all_captions, handle, protocol=pickle.HIGHEST_PROTOCOL)
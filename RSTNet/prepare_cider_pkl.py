import json
import sys
sys.path.append('../PureT')
from scorer.ciderD_scorer import CiderScorer
import tqdm
import pickle
def get_doc_freq(refs):
    tmp = CiderScorer(df_mode="corpus")
    for ref in refs:
        tmp.cook_append(None, ref)
    tmp.compute_doc_freq()
    return tmp.document_frequency, len(tmp.crefs)

def build_dict(imgs, wtoi):
    wtoi['<eos>'] = 0
    count_imgs = 0

    refs_words = []
    refs_idxs = []
    for img in imgs:
        ref_words = []
        ref_idxs = []
        for sent in img['sentences']:
            tmp_tokens = sent['tokens'] + ['<eos>']
            tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
            ref_words.append(' '.join(tmp_tokens))
            ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
        refs_words.append(ref_words)
        refs_idxs.append(ref_idxs)
        count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words, count_refs = get_doc_freq(refs_words)
    ngram_idxs, count_refs = get_doc_freq(refs_idxs)
    print('count_refs:', count_refs)
    return ngram_words, ngram_idxs, count_refs

stoi_dict_vocab = json.load(open('../dataset/stoi_dict_vocab.json'))
dict_imgs = json.load(open("../dataset/dict_imgs.json", "r"))
imgs = [dict_imgs[key] for key in dict_imgs]
ngram_words, ngram_idxs, ref_len = build_dict(imgs, stoi_dict_vocab)
c = 0
for key in ngram_words:
    if ngram_words[key] > 1:
        c+=1
        print(key, ngram_words[key])
print(c)
with open('../dataset/dacoisamsung_cider.pkl', 'wb') as handle: pickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, handle, protocol=pickle.HIGHEST_PROTOCOL)
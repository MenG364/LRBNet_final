import json
import pickle

import h5py
import joblib
import torch
from tqdm import tqdm

import utils
from dataset import Dictionary
from os.path import join

feat=joblib.load(open('F:\datasets\VQA\VQA2.0\dense_train_feat.pkl','rb'))
dictionary = Dictionary.load_from_file(
    join('F:\datasets\VQA\VQA2.0', 'glove/dictionary.pkl'))


def cap_tokenize(captions, max_length=20):
    """Tokenizes the questions.

    This will add q_token in each entry of the dataset.
    -1 represent nil, and should be treated as padding_idx in embedding
    """
    caps = []
    for entry in captions:
        tokens = dictionary.tokenize(entry, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad to the back of the sentence
            padding = [dictionary.padding_idx] * \
                      (max_length - len(tokens))
            tokens = tokens + padding
        utils.assert_eq(len(tokens), max_length)
        caps.append(tokens)
    return caps


cap_file = 'E:\PycharmProjects\densecap\output\\train_cap.json'

cap = json.load(open(cap_file, 'r'))
cap = {int(key.split('/')[-1].split('.')[0].split('_')[-1]): cap[key] for key in cap.keys()}
captions = {}
with tqdm(total=len(cap)) as t:
    for key in cap.keys():
        caption = [[],[]]
        caption[0] = key
        caption[1] = [cap['cap'] for cap in cap[key]]
        caption[1] = cap_tokenize(caption[1], max_length=17)
        caption[1] = torch.as_tensor(caption[1])
        captions[key]=caption
        t.update(1)

feat_file = 'E:\PycharmProjects\densecap\output\\train_feat.h5'
feat_map = 'E:\PycharmProjects\densecap\output\\train_feat_img_mappings.txt'
map=[]
with open(feat_map, 'r') as f:
    map=f.readlines()
features = {}
bbs = {}
with h5py.File(feat_file, 'r') as feat:
    with tqdm(total=len(map)) as t:
        for i in range(len(map)):
            feature = [[],[],[]]
            bb = [[],[]]
            img_id = int(map[i].split('.')[0].split('_')[-1])
            feature[0] = img_id
            bb[0] = img_id
            start_idx = feat['start_idx'][i]
            end_idx = feat['end_idx'][i]+1
            feature[1] = end_idx - start_idx + 1
            feature[2] = feat['feats'][start_idx: end_idx]
            feature[2] = torch.from_numpy(feature[2])
            bb[1] = feat['boxes'][start_idx: end_idx]
            bbs[img_id]=bb
            features[img_id]=feature
            t.update(1)

img_id2idx = pickle.load(open(join('F:\datasets\VQA\VQA2.0/imgids', 'train_imgid2idx.pkl'), 'rb'))
features=[features[key] for key in img_id2idx]
captions=[captions[key] for key in img_id2idx]
bbs=[bbs[key] for key in img_id2idx]

dataset_file = 'data/dense_train_feat.pkl'
joblib.dump((features,captions,bbs), open(dataset_file, 'wb'))

print()

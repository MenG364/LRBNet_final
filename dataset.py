import base64
import itertools
import json
import os
import pickle
import h5py
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as op

import tools.compute_softscore
import utils
from tsv_scripts.misc import load_from_yaml_file, find_file_path_in_yaml
from tsv_scripts.tsv_file import TSVFile
from vqa_utils import VqaUtils

COUNTING_ONLY = False


# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
            ('number of' in q.lower() and 'number of the' not in q.lower()) or \
            'amount of' in q.lower() or \
            'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '') \
            .replace('?', '').replace('\'s', ' \'s').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK
                # for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None != answer and 'image_id' in answer.keys():
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer,
        'type': question['types'] if 'types' in question.keys() else 'None'
    }
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """

    question_path = os.path.join(
        dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
                  (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # train, val
    if 'test' != name[:4]:
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY \
                    or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))

    # test2015
    else:
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY \
                    or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries


def _load_dataset_v1(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """

    question_path = os.path.join(
        dataroot, 'Questions/OpenEnded_mscoco_%s_questions.json' %
                  (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # train, val
    if 'test' != name[:4]:
        answer_path = os.path.join(dataroot, 'cache', '%s_target_v1.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY \
                    or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))

    # test2015
    else:
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY \
                    or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries


def _load_GQA(dataroot, name, img_id2val, label2ans, banlance=False):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev', test','challenge'
    """
    if name == 'submission':
        banlance = False
    if banlance == False:
        question_path = os.path.join(dataroot,
                                     'Questions/all/%s_all_questions.pkl' % (name if 'test' != name[:4] else name))
    else:
        question_path = os.path.join(dataroot,
                                     'Questions/balanced/%s_balanced_questions.pkl' % (
                                         name if 'test' != name[:4] else name))
    questions = pickle.load(open(question_path, 'rb'))
    q = sorted(questions.items())
    for ques in q:
        ques[1]['question_id'] = ques[0]
    questions = [ques[1] for ques in q]
    # train, val
    if 'test' != name and 'submission' != name:
        if name == "testdev":
            ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
            ans2label = pickle.load(open(ans2label_path, 'rb'))
            answers = [{'question_id': ques['question_id'],
                        'labels': ans2label[ques['answer']] if ques['answer'] in ans2label.keys() else 0} for ques in
                       questions]
        else:
            if banlance == False:
                answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
            else:
                answer_path = os.path.join(dataroot, 'cache', '%s_balanced_target.pkl' % name)
            answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            # utils.assert_eq(question['answer'], label2ans[answer['labels']])
            # utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            # if img_id2val[img_id] <= 4000:
            if not COUNTING_ONLY \
                    or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))

    # test2015
    else:
        entries = []
        for question in questions:
            img_id = question['image_id']
            # if img_id2val[img_id] <= 4000:
            if not COUNTING_ONLY \
                    or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries


def _load_CLEVR(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(dataroot, 'Questions/%s_questions.json' % (name if 'test' != name[:4] else name))
    questions = json.load(open(question_path, 'rb'))['questions']
    # train, val
    if 'test' != name[:4]:
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)

        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            answer['labels'] = answer['labels'][0]
            # utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = name + '_%06d' % answer['image_id']
            # if img_id2val[img_id] <= 4000:
            if not COUNTING_ONLY \
                    or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))

    # test2015
    else:
        entries = []
        for question in questions:
            img_id = name + '_%06d' % question['image_id']
            if not COUNTING_ONLY \
                    or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries


def _load_visualgenome(dataroot, name, img_id2val, label2ans, adaptive=True):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(dataroot,
                                 'visualGenome/question_answers.json')
    image_data_path = os.path.join(dataroot,
                                   'visualGenome/image_data.json')
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    cache_path = os.path.join(dataroot, 'cache', 'vg_%s%s_target.pkl' %
                              (name, '_adaptive' if adaptive else ''))

    if os.path.isfile(cache_path):
        entries = pickle.load(open(cache_path, 'rb'))
    else:
        entries = []
        ans2label = pickle.load(open(ans2label_path, 'rb'))
        vgq = json.load(open(question_path, 'r'))
        _vgv = json.load(open(image_data_path, 'r'))
        vgv = {}
        for _v in _vgv:
            if _v['coco_id']:
                vgv[_v['image_id']] = _v['coco_id']
        # used image, used question, total question, out-of-split
        counts = [0, 0, 0, 0]
        for vg in vgq:
            coco_id = vgv.get(vg['id'], None)
            if coco_id is not None:
                counts[0] += 1
                img_idx = img_id2val.get(coco_id, None)
                if img_idx is None:
                    counts[3] += 1
                for q in vg['qas']:
                    counts[2] += 1
                    _answer = tools.compute_softscore.preprocess_answer(
                        q['answer'])
                    label = ans2label.get(_answer, None)
                    if label and img_idx:
                        counts[1] += 1
                        answer = {
                            'labels': [label],
                            'scores': [1.]}
                        entry = {
                            'question_id': q['qa_id'],
                            'image_id': coco_id,
                            'image': img_idx,
                            'question': q['question'],
                            'answer': answer,
                            'type': q['types'] if 'types' in q.keys() else 'None'}
                        if not COUNTING_ONLY \
                                or is_howmany(q['question'], answer, label2ans):
                            entries.append(entry)

        print('Loading VisualGenome %s' % name)
        print('\tUsed COCO images: %d/%d (%.4f)' %
              (counts[0], len(_vgv), counts[0] / len(_vgv)))
        print('\tOut-of-split COCO images: %d/%d (%.4f)' %
              (counts[3], counts[0], counts[3] / counts[0]))
        print('\tUsed VG questions: %d/%d (%.4f)' %
              (counts[1], counts[2], counts[1] / counts[2]))
        with open(cache_path, 'wb') as f:
            pickle.dump(entries, open(cache_path, 'wb'))

    return entries


def _load_visualgenome_v1(dataroot, name, img_id2val, label2ans, adaptive=True):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(dataroot,
                                 'visualGenome/question_answers.json')
    image_data_path = os.path.join(dataroot,
                                   'visualGenome/image_data.json')
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label_v1.pkl')
    cache_path = os.path.join(dataroot, 'cache', 'vg_%s%s_target.pkl' %
                              (name, '_adaptive_v1' if adaptive else ''))

    if os.path.isfile(cache_path):
        entries = pickle.load(open(cache_path, 'rb'))
    else:
        entries = []
        ans2label = pickle.load(open(ans2label_path, 'rb'))
        vgq = json.load(open(question_path, 'r'))
        _vgv = json.load(open(image_data_path, 'r'))
        vgv = {}
        for _v in _vgv:
            if _v['coco_id']:
                vgv[_v['image_id']] = _v['coco_id']
        # used image, used question, total question, out-of-split
        counts = [0, 0, 0, 0]
        for vg in vgq:
            coco_id = vgv.get(vg['id'], None)
            if coco_id is not None:
                counts[0] += 1
                img_idx = img_id2val.get(coco_id, None)
                if img_idx is None:
                    counts[3] += 1
                for q in vg['qas']:
                    counts[2] += 1
                    _answer = tools.compute_softscore.preprocess_answer(
                        q['answer'])
                    label = ans2label.get(_answer, None)
                    if label and img_idx:
                        counts[1] += 1
                        answer = {
                            'labels': [label],
                            'scores': [1.]}
                        entry = {
                            'question_id': q['qa_id'],
                            'image_id': coco_id,
                            'image': img_idx,
                            'question': q['question'],
                            'answer': answer,
                            'type': q['types'] if 'types' in q.keys() else 'None'}
                        if not COUNTING_ONLY \
                                or is_howmany(q['question'], answer, label2ans):
                            entries.append(entry)

        print('Loading VisualGenome %s' % name)
        print('\tUsed COCO images: %d/%d (%.4f)' %
              (counts[0], len(_vgv), counts[0] / len(_vgv)))
        print('\tOut-of-split COCO images: %d/%d (%.4f)' %
              (counts[3], counts[0], counts[3] / counts[0]))
        print('\tUsed VG questions: %d/%d (%.4f)' %
              (counts[1], counts[2], counts[1] / counts[2]))
        with open(cache_path, 'wb') as f:
            pickle.dump(entries, open(cache_path, 'wb'))

    return entries


def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['image_id'] == vgv_id:
            return v['coco_id']
    return None


def get_question_id_to_question_type(annotations):
    qid_to_qtype = {}
    qid_to_atype = {}
    if 'annotations' in annotations:
        annotations = annotations['annotations']
    for ann in annotations:
        qid_to_qtype[str(ann['question_id'])] = ann['question_type']
        qid_to_atype[str(ann['question_id'])] = ann['answer_type']
    return qid_to_qtype, qid_to_atype


class VQAFeatureDataset(Dataset):
    def __init__(self, dataset, name, dictionary, relation_type, adaptive=False, dataroot='./data',
                 is_train=True, pos_emb_dim=64, nongt_dim=36, **kwargs):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test2015', 'test-dev2015', 'test', 'testdev', 'challenge']

        self.root = dataroot
        self.dataset = dataset

        dataroot = self.root
        ans2label_path = os.path.join(self.root, 'cache',
                                      'trainval_ans2label.pkl' if not self.dataset == "vqa1.0" else 'trainval_ans2label_v1.pkl')
        label2ans_path = os.path.join(self.root, 'cache',
                                      'trainval_label2ans.pkl' if not self.dataset == "vqa1.0" else 'trainval_label2ans_v1.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))

        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.relation_type = relation_type
        self.adaptive = adaptive
        prefix = '36'
        if 'test' in name:
            prefix = '_36'

        imgid_dataroot = dataroot + "/imgids"
        if self.dataset == 'gqa' and name in ['test', 'testdev', 'challenge']:
            if name == 'test':
                name = 'submission'
            self.img_id2idx = pickle.load(open(os.path.join(imgid_dataroot, 'submission_imgid2idx.pkl'), 'rb'))
            self.h5_path = os.path.join(dataroot, 'features_adaptive', 'dense_submission.hdf5')
        else:
            if name == 'test-dev2015' or name == 'test2015':
                self.img_id2idx = pickle.load(open(os.path.join(imgid_dataroot, '%s_imgid2idx.pkl' % ("test")), 'rb'))
                self.h5_path = os.path.join(dataroot, 'features_adaptive', 'dense_%s%s.hdf5' %
                                            ("test", '' if self.adaptive else prefix))
            else:
                self.img_id2idx = pickle.load(open(os.path.join(imgid_dataroot, '%s_imgid2idx.pkl' % (name)), 'rb'))
                self.h5_path = os.path.join(dataroot, 'features_adaptive', 'dense_%s%s.hdf5' %
                                            (name, '' if self.adaptive else prefix))

        print('loading features from h5 file %s' % self.h5_path)

        hf = h5py.File(self.h5_path, 'r')
        normalized_bb = hf.get('spatial_features')
        max_length = 20
        if self.dataset == 'gqa':
            max_length = 29
            self.entries = _load_GQA(dataroot, name, self.img_id2idx, self.label2ans, banlance=True)
        elif self.dataset == 'vqa':
            max_length = 20
            self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)
        elif self.dataset == 'vqa1.0':
            max_length = 20
            self.entries = _load_dataset_v1(dataroot, name, self.img_id2idx, self.label2ans)
        elif self.dataset == 'clevr':
            max_length = 45
            self.entries = _load_CLEVR(dataroot, name, self.img_id2idx, self.label2ans)
        self.tokenize(max_length)

        self.tensorize()
        self.nongt_dim = nongt_dim
        self.emb_dim = pos_emb_dim
        self.v_dim = 2054
        self.s_dim = normalized_bb[(0)].size
        self.load_h5()

    def close_h5_file(self):
        try:
            self.hf.close()
        except:
            pass

    def tokenize(self, max_length=20):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad to the back of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                if (self.dataset == 'vqa' or self.dataset == 'vqa1.0'):
                    scores = np.array(answer['scores'], dtype=np.float32)
                else:
                    scores = np.array(1, dtype=np.float32)
                if labels.size > 0:
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def load_h5(self):
        if not hasattr(self, 'hf'):
            self.hf = h5py.File(self.h5_path, 'r')
            self.features = self.hf.get('image_features')
            self.normalized_bb = self.hf.get('spatial_features')
            self.bb = self.hf.get('image_bb')
            self.captions = self.hf.get('captions')
            self.pos_boxes = self.hf.get('pos_boxes')
            self.objects_num = self.hf.get('objects_num')

    def __getitem__(self, index):
        self.load_h5()

        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["image_id"]
        image_idx = entry["image"]
        question = entry['q_token']
        question_id = entry['question_id']
        question_type = entry['type']

        start_idx = self.pos_boxes[image_idx][0]
        end_idx = self.pos_boxes[image_idx][1]
        features = torch.from_numpy(self.features[start_idx:end_idx, :])
        normalized_bb = torch.from_numpy(self.normalized_bb[start_idx:end_idx, :])
        bb = torch.from_numpy(self.bb[start_idx:end_idx, :])
        captions = torch.from_numpy(self.captions[start_idx:end_idx, :17])
        # if self.dataset=='clevr':
        #     captions=torch.cat((captions,torch.tensor(19901).repeat(captions.size(0),13)),-1)
        objects_num = self.objects_num[image_idx]
        utils.assert_eq(str(len(features)), str(objects_num))
        utils.assert_eq(str(len(normalized_bb)), str(objects_num))
        utils.assert_eq(str(len(bb)), str(objects_num))
        utils.assert_eq(str(len(captions)), str(objects_num))

        answer = entry['answer']
        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            target = labels
            if (self.dataset == 'vqa' or self.dataset == 'vqa1.0'):
                target = torch.zeros(self.num_ans_candidates)
                if labels is not None:
                    labels = labels.to(torch.long)
                    target.scatter_(0, labels, scores)
            return features, normalized_bb, question, target, captions, \
                   question_type, question_id, image_id, bb

        else:
            return features, normalized_bb, question, captions, \
                   question_type, question_id, image_id, bb

    def __len__(self):
        return len(self.entries)


class VisualGenomeFeatureDataset(Dataset):
    def __init__(self, dataset, name, features, normalized_bb, bb, dictionary, captions, objects_num, dataroot='data',
                 adaptive=False,
                 pos_boxes=None, pos_emb_dim=64):
        super(VisualGenomeFeatureDataset, self).__init__()
        # do not use test split images!
        assert name in ['train', 'val']
        print('loading Visual Genome data %s' % name)
        self.dataset = dataset
        ans2label_path = os.path.join(dataroot, 'cache',
                                      'trainval_ans2label.pkl' if not self.dataset == "vqa1.0" else 'trainval_ans2label_v1.pkl')
        label2ans_path = os.path.join(dataroot, 'cache',
                                      'trainval_label2ans.pkl' if not self.dataset == "vqa1.0" else 'trainval_label2ans_v1.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, 'imgids/%s%s_imgid2idx.pkl' %
                              (name, '' if self.adaptive else '36')),
                 'rb'))
        self.bb = bb
        self.features = features
        self.normalized_bb = normalized_bb
        self.captions = captions
        self.objects_num = objects_num

        if self.adaptive:
            self.pos_boxes = pos_boxes

        if self.dataset == 'vqa':
            self.entries = _load_visualgenome(dataroot, name, self.img_id2idx,
                                              self.label2ans,
                                              adaptive=self.adaptive)
        elif self.dataset == 'vqa1.0':
            self.entries = _load_visualgenome_v1(dataroot, name, self.img_id2idx,
                                                 self.label2ans,
                                                 adaptive=self.adaptive)
        self.tokenize()
        self.tensorize()
        self.emb_dim = pos_emb_dim
        self.v_dim = 2054
        self.s_dim = normalized_bb[(0)].size

    def tokenize(self, max_length=20):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            entry['q_len'] = len(tokens)
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["image_id"]
        image_idx = entry["image"]
        question = entry['q_token']
        question_id = entry['question_id']
        question_type = entry['type']

        start_idx = self.pos_boxes[image_idx][0]
        end_idx = self.pos_boxes[image_idx][1]
        features = torch.from_numpy(self.features[start_idx:end_idx, :])
        normalized_bb = torch.from_numpy(self.normalized_bb[start_idx:end_idx, :])
        bb = torch.from_numpy(self.bb[start_idx:end_idx, :])
        captions = torch.from_numpy(self.captions[start_idx:end_idx, :17])
        # if self.dataset=='clevr':
        #     captions=torch.cat((captions,torch.tensor(19901).repeat(captions.size(0),13)),-1)
        objects_num = self.objects_num[image_idx]
        utils.assert_eq(str(len(features)), str(objects_num))
        utils.assert_eq(str(len(normalized_bb)), str(objects_num))
        utils.assert_eq(str(len(bb)), str(objects_num))
        utils.assert_eq(str(len(captions)), str(objects_num))

        answer = entry['answer']
        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            # target = labels
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                labels = labels.to(torch.long)
                target.scatter_(0, labels, scores)
            return features, normalized_bb, question, target, captions, \
                   question_type, question_id, image_id, bb

        else:
            return features, normalized_bb, question, captions, \
                   question_type, question_id, image_id, bb

    def __len__(self):
        return len(self.entries)


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70,
                 max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
                 is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len,
                                                     self.max_seq_len), dtype=torch.long))


def tfidf_from_questions(names, dictionary, dataroot='data',
                         target=['vqa', 'vg']):
    # rows, cols for uncoalesce sparse matrix
    inds = [[], []]
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0])
                inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1])
                inds[1].append(c[0])

    # VQA 2.0
    if 'vqa' in target:
        for name in names:
            assert name in ['train', 'val', 'test-dev2015', 'test2015']
            question_path = os.path.join(
                dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
                          (name + '2014' if 'test' != name[:4] else name))
            questions = json.load(open(question_path))['questions']

            for question in questions:
                populate(inds, df, question['question'])

    # Visual Genome
    if 'vg' in target:
        question_path = os.path.join(dataroot, 'visualGenome',
                                     'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])

    # TF-IDF
    vals = np.ones((len(inds[1])))
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds),
                                     torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = dataroot + '/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(
        dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0),
                                                                tfidf.size(1)))

    return tfidf, weights

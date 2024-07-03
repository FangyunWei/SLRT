import torch
from modelling.two_stream import S3D_two_stream, S3D_two_stream_v2
from modelling.S3D import S3D_backbone
from modelling.pyramid import PyramidNetwork
# from modelling.mae import MAE_S3D
from utils.misc import make_logger
from utils.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014
import gzip, pickle
from collections import defaultdict
import pandas as pd
import unicodedata
import os
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from utils.langconv import Converter
import matplotlib.pyplot as plt
import cv2
import random; random.seed(0)
from utils.metrics import bleu, rouge, wer_list, wer_list_per_sen
from utils.external_metrics.sacrebleu import sentence_bleu
from opencc import OpenCC


def clean_tvb(s):
    op = []
    for t in s.split():
        if '<' in t and '>' in t:
            continue
        op.append(t)
    return ' '.join(op)


if __name__ == '__main__':
    def clean(s):
        a = []
        for item in s:
            if item != '':
                a.append(item)
        return a


    split = 'train'
    path = f'../../data/tvb/split/v5.7/{split}.csv'
    root = '../../data/tvb/grouped/sign'
    df = pd.read_csv(path, sep="|")
    # df = df.dropna()

    # words = df["words"].apply(lambda s: unicodedata.normalize("NFKC", s))
    # words = words.apply(list)
    # df["words"] = words

    # glosses = df["glosses"].apply(lambda s: unicodedata.normalize("NFKC", s))
    # # glosses = df["glosses"].apply(lambda s: s.strip())
    # glosses = glosses.str.split("[ +]")
    # df["glosses"] = glosses
    # df["glosses"] = df["glosses"].apply(lambda s: clean(s))

    # data = []
    # name_lst = []
    # for i, row in df.iterrows():
    #     # print(row['glosses'])
    #     name = row['id']
    #     # gloss = ' '.join(row['glosses'])
    #     # gloss = gloss.split()
    #     # gloss = ' '.join(gloss)
    #     # text = ' '.join(row['words'])
    #     # text = text.split()
    #     # text = ' '.join(text)
    #     num_frames = len(os.listdir(os.path.join(root, name)))
    #     # data.append({'name': name, 'gloss': gloss, 'text': text, 'num_frames': num_frames, 'signer': row['signer']})
    #     data.append({'name': name, 'gloss': '1', 'text': '1', 'num_frames': num_frames, 'signer': row['signer']})
    #     name_lst.append(name)
    # print(len(data))

    # with open('../../data/tvb/v5.6_train.pkl', 'rb') as f:
    #     train = pickle.load(f)
    # # with open('../../data/tvb/v5.6_dev.pkl', 'rb') as f:
    # #     dev = pickle.load(f)
    # # with open('../../data/tvb/v5.6_test.pkl', 'rb') as f:
    # #     test = pickle.load(f)

    # train_name = [item['name'] for item in train]
    # # test_name = [item['name'] for item in test]

    # remain = []
    # for item in data:
    #     if item['name'] not in train_name:
    #         remain.append(item)
    # print(len(remain))
    # with open('../../data/tvb/v5.6_train_remain.pkl', 'wb') as f:
    #     pickle.dump(remain, f)

    # gloss2ids = {'<s>': 0, '<unk>': 1, '<pad>': 2, '</s>': 3}
    # i = 4
    # with open('../../data/tvb/split/v5.6/vocab.txt', 'r') as f:
    #     vocab = f.readlines()
    #     for v in vocab:
    #         v = v.strip()
    #         gloss2ids[v] = i
    #         i += 1
    
    # for item in [*train, *dev, *test]:
    #     for g in item['gloss'].split():
    #         if g not in gloss2ids:
    #             print(g)
    # with open('../../data/tvb/v5.6_gloss2ids.pkl', 'wb') as f:
    #     pickle.dump(gloss2ids, f)

    # for item in train:
    #     num_frames = item['num_frames']
    #     name = item['name']
    #     k = kps[name]['keypoints']
    #     if k.shape[0] != num_frames:
    #         print(name, k.shape[0], num_frames)

    # with open('../../data/tvb/keypoints_hrnet_dark_coco_wholebody.pkl', 'rb') as f:
    #     kps = pickle.load(f)
    # with open('../../data/tvb/keypoints_hrnet_dark_coco_wholebody/train.pkl', 'rb') as f:
    #     train_r = pickle.load(f)
    # # with open('../../data/tvb/keypoints_hrnet_dark_coco_wholebody/test.pkl', 'rb') as f:
    # #     test_r = pickle.load(f)
    
    # for name, kp in train_r.items():
    #     kps[name] = {'keypoints': kp}
    # # for name, kp in test_r.items():
    # #     kps[name] = {'keypoints': kp}
    # with open('../../data/tvb/keypoints_hrnet_dark_coco_wholebody.pkl', 'wb') as f:
    #     print(len(kps))
    #     pickle.dump(kps, f)


    # with open('../../data/tvb/keypoints_hrnet_dark_coco_wholebody/train.pkl', 'rb') as f:
    #     remain = pickle.load(f)
    # for k,v in remain.items():
    #     assert k in kps
    #     kps[k] = {'keypoints': v}
    # with open('../../data/tvb/keypoints_hrnet_dark_coco_wholebody.pkl', 'wb') as f:
    #     pickle.dump(kps, f)

    # with open('../../data/tvb/v5.6_train.pkl', 'rb') as f:
    #     train = pickle.load(f)
    # train2items = {item['name']: item for item in train}
    # with open('../../data/tvb/v5.6_dev.pkl', 'rb') as f:
    #     dev = pickle.load(f)
    # dev2items = {item['name']: item for item in dev}
    # with open('../../data/tvb/v5.6_test.pkl', 'rb') as f:
    #     test = pickle.load(f)
    # test2items = {item['name']: item for item in test}

    # for n in ['signer0', 'signer0_75', 'signer0_50', 'signer0_25']:
    #     fname = '../../data/tvb/train_{}.pkl'.format(n)
    #     with open(fname, 'rb') as f:
    #         data = pickle.load(f)
    #     new_data = []
    #     for item in data:
    #         new_data.append(train2items[item['name']])
    #     with open('../../data/tvb/v5.6_train_{}.pkl'.format(n), 'wb') as f:
    #         pickle.dump(new_data, f)

    # fname = '../../data/tvb/dev_signer0.pkl'
    # with open(fname, 'rb') as f:
    #     data = pickle.load(f)
    # new_data = []
    # for item in data:
    #     new_data.append(dev2items[item['name']])
    # with open('../../data/tvb/v5.6_dev_signer0.pkl', 'wb') as f:
    #     pickle.dump(new_data, f)

    # fname = '../../data/tvb/test_signer0.pkl'
    # with open(fname, 'rb') as f:
    #     data = pickle.load(f)
    # new_data = []
    # for item in data:
    #     new_data.append(test2items[item['name']])
    # with open('../../data/tvb/v5.6_test_signer0.pkl', 'wb') as f:
    #     pickle.dump(new_data, f)
    
    #--------------------------------------------simplified------------------------------------------------
    def Traditional2Simplified(sentence):
    # '''
    # 将sentence中的繁体字转为简体字
    # :param sentence: 待转换的句子
    # :return: 将句子中繁体字转换为简体字之后的句子
    # '''
        sentence = Converter('zh-hans').convert(sentence)
        return sentence
    
    def Simplified2Traditional(sentence):
    # '''
    # 将sentence中的繁体字转为简体字
    # :param sentence: 待转换的句子
    # :return: 将句子中繁体字转换为简体字之后的句子
    # '''
        sentence = Converter('zh-hant').convert(sentence)
        return sentence

    print(Traditional2Simplified('這裡'))

    # with open('../../data/tvb/v5.6_train_signer0_75.pkl', 'rb') as f:
    #     train = pickle.load(f)
    # with open('../../data/tvb/v5.6_train_signer0_50.pkl', 'rb') as f:
    #     dev = pickle.load(f)
    # with open('../../data/tvb/v5.6_train_signer0_25.pkl', 'rb') as f:
    #     test = pickle.load(f)
    
    # new_train = []
    # for item in train:
    #     new_item = deepcopy(item)
    #     text = item['text']
    #     new_text = ''.join(text.strip().split())
    #     assert ' ' not in new_text
    #     new_text = Traditional2Simplified(new_text)
    #     new_gloss = Traditional2Simplified(item['gloss'])
    #     new_item['text'] = new_text
    #     new_item['gloss'] = new_gloss
    #     new_train.append(new_item)

    # new_dev = []
    # for item in dev:
    #     new_item = deepcopy(item)
    #     text = item['text']
    #     new_text = ''.join(text.strip().split())
    #     assert ' ' not in new_text
    #     new_text = Traditional2Simplified(new_text)
    #     new_gloss = Traditional2Simplified(item['gloss'])
    #     new_item['text'] = new_text
    #     new_item['gloss'] = new_gloss
    #     new_dev.append(new_item)

    # new_test = []
    # for item in test:
    #     new_item = deepcopy(item)
    #     text = item['text']
    #     new_text = ''.join(text.strip().split())
    #     assert ' ' not in new_text
    #     new_text = Traditional2Simplified(new_text)
    #     new_gloss = Traditional2Simplified(item['gloss'])
    #     print(new_text)
    #     print(new_gloss)
    #     new_item['text'] = new_text
    #     new_item['gloss'] = new_gloss
    #     new_test.append(new_item)

    # with open('../../data/tvb/v5.6_train_signer0_75_sim.pkl', 'wb') as f:
    #     pickle.dump(new_train, f)
    # with open('../../data/tvb/v5.6_train_signer0_50_sim.pkl', 'wb') as f:
    #     pickle.dump(new_dev, f)
    # with open('../../data/tvb/v5.6_train_signer0_25_sim.pkl', 'wb') as f:
    #     pickle.dump(new_test, f)

    # with open('../../data/tvb/v5.6_gloss2ids.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # new = {}
    # for k,v in data.items():
    #     new_k = Traditional2Simplified(k)
    #     if new_k in new or k=='恆生':
    #         new[k] = v
    #         print(new_k, k)
    #     else:
    #         new[new_k] = v
    # with open('../../data/tvb/v5.6_gloss2ids_sim.pkl', 'wb') as f:
    #     print(len(new))
    #     pickle.dump(new, f)
    #--------------------------------------------simplified------------------------------------------------


    #--------------------------------------------signer oov-------------------------------------------------
    # with open('../../data/tvb/v5.6_train_signer0_25.pkl', 'rb') as f:
    #     train_25 = pickle.load(f)
    # with open('../../data/tvb/v5.6_train_signer0_50.pkl', 'rb') as f:
    #     train_50 = pickle.load(f)
    # with open('../../data/tvb/v5.6_train_signer0_75.pkl', 'rb') as f:
    #     train_75 = pickle.load(f)
    # with open('../../data/tvb/v5.6_train_signer0.pkl', 'rb') as f:
    #     train = pickle.load(f)
    # with open('../../data/tvb/v5.7_dev_signer0.pkl', 'rb') as f:
    #     dev = pickle.load(f)
    # with open('../../data/tvb/v5.7_test_signer0.pkl', 'rb') as f:
    #     test = pickle.load(f)

    # oov_dev = []
    # oov_test = []

    # gloss2times_dev = {}
    # gloss2times_test = {}
    # for item in dev:
    #     gloss = item['gloss'].split()  #item['gloss'].split()
    #     for g in gloss:
    #         gloss2times_dev[g] = gloss2times_dev.get(g, 0) + 1
    # for item in test:
    #     gloss = item['gloss'].split()  #item['gloss'].split()
    #     for g in gloss:
    #         gloss2times_test[g] = gloss2times_test.get(g, 0) + 1

    # vocab = []
    # vocab_25 = []
    # vocab_50 = []
    # vocab_75 = []
    # for item in train:
    #     for g in item['gloss'].split():  #item['gloss'].split():
    #         if g not in vocab:
    #             vocab.append(g)
    # for item in train_25:
    #     for g in item['gloss'].split():  #item['gloss'].split():
    #         if g not in vocab_25:
    #             vocab_25.append(g)
    # for item in train_50:
    #     for g in item['gloss'].split():  #item['gloss'].split():
    #         if g not in vocab_50:
    #             vocab_50.append(g)
    # for item in train_75:
    #     for g in item['gloss'].split():  #item['gloss'].split():
    #         if g not in vocab_75:
    #             vocab_75.append(g)
    
    # gloss2times = gloss2times_test
    # vocab_test = list(gloss2times.keys())
    # vocab_diff_25 = set(vocab_test) - (set(vocab_25) & set(vocab_test))
    # count = 0
    # for g in vocab_diff_25:
    #     count += gloss2times[g]
    # print('size_25: ', len(vocab_25))
    # print('oov_25: ', 100*count/sum(gloss2times.values()))
    # vocab_diff_50 = set(vocab_test) - (set(vocab_50) & set(vocab_test))
    # count = 0
    # for g in vocab_diff_50:
    #     count += gloss2times[g]
    # print('size_50: ', len(vocab_50))
    # print('oov_50: ', 100*count/sum(gloss2times.values()))
    # vocab_diff_75 = set(vocab_test) - (set(vocab_75) & set(vocab_test))
    # count = 0
    # for g in vocab_diff_75:
    #     count += gloss2times[g]
    # print('size_75: ', len(vocab_75))
    # print('oov_75: ', 100*count/sum(gloss2times.values()))
    # vocab_diff_100 = set(vocab_test) - (set(vocab) & set(vocab_test))
    # count = 0
    # for g in vocab_diff_100:
    #     count += gloss2times[g]
    # print('size_100: ', len(vocab))
    # print('oov_100: ', 100*count/sum(gloss2times.values()))
    # print('size_test_vocab: ', len(vocab_test))
    # --------------------------------------------signer oov-------------------------------------------------

    #----------------------------------------------text update-----------------------------------------------
    # split = 'dev'
    # path = f'../../data/tvb/split/v5.6/{split}.csv'
    # root = '../../data/tvb/grouped/sign'
    # df = pd.read_csv(path, sep="|")
    # df = df.dropna()

    # words = df["words"].apply(lambda s: unicodedata.normalize("NFKC", s))
    # words = words.apply(list)
    # df["words"] = words

    # glosses = df["glosses"].apply(lambda s: unicodedata.normalize("NFKC", s))
    # # glosses = df["glosses"].apply(lambda s: s.strip())
    # glosses = glosses.str.split("[ +]")
    # df["glosses"] = glosses
    # df["glosses"] = df["glosses"].apply(lambda s: clean(s))

    # data = []
    # name_lst = []
    # for i, row in df.iterrows():
    #     # print(row['glosses'])
    #     name = row['id']
    #     gloss = ' '.join(row['glosses'])
    #     gloss = gloss.split()
    #     gloss = ' '.join(gloss)
    #     text = ' '.join(row['words'])
    #     text = text.split()
    #     text = ''.join(text)
    #     num_frames = len(os.listdir(os.path.join(root, name)))
    #     data.append({'name': name, 'gloss': gloss, 'text': text, 'num_frames': num_frames})
    #     name_lst.append(name)
    # print(len(data))
    # name2text_dev = {item['name']: item['text'] for item in data}

    # split = 'test'
    # path = f'../../data/tvb/split/v5.6/{split}.csv'
    # root = '../../data/tvb/grouped/sign'
    # df = pd.read_csv(path, sep="|")
    # df = df.dropna()

    # words = df["words"].apply(lambda s: unicodedata.normalize("NFKC", s))
    # words = words.apply(list)
    # df["words"] = words

    # glosses = df["glosses"].apply(lambda s: unicodedata.normalize("NFKC", s))
    # # glosses = df["glosses"].apply(lambda s: s.strip())
    # glosses = glosses.str.split("[ +]")
    # df["glosses"] = glosses
    # df["glosses"] = df["glosses"].apply(lambda s: clean(s))

    # data = []
    # name_lst = []
    # for i, row in df.iterrows():
    #     # print(row['glosses'])
    #     name = row['id']
    #     gloss = ' '.join(row['glosses'])
    #     gloss = gloss.split()
    #     gloss = ' '.join(gloss)
    #     text = ' '.join(row['words'])
    #     text = text.split()
    #     text = ''.join(text)
    #     num_frames = len(os.listdir(os.path.join(root, name)))
    #     data.append({'name': name, 'gloss': gloss, 'text': text, 'num_frames': num_frames})
    #     name_lst.append(name)
    # print(len(data))
    # name2text_test = {item['name']: item['text'] for item in data}

    # with open('../../data/tvb/v5.6_dev.pkl', 'rb') as f:
    #     dev = pickle.load(f)
    # with open('../../data/tvb/v5.6_test.pkl', 'rb') as f:
    #     test = pickle.load(f)
    # with open('../../data/tvb/v5.6_dev_signer0.pkl', 'rb') as f:
    #     dev_s0 = pickle.load(f)
    # with open('../../data/tvb/v5.6_test_signer0.pkl', 'rb') as f:
    #     test_s0 = pickle.load(f)
    # with open('../../data/tvb/v5.6_dev_sim.pkl', 'rb') as f:
    #     dev_sim = pickle.load(f)
    # with open('../../data/tvb/v5.6_test_sim.pkl', 'rb') as f:
    #     test_sim = pickle.load(f)
    # with open('../../data/tvb/v5.6_dev_signer0_sim.pkl', 'rb') as f:
    #     dev_s0_sim = pickle.load(f)
    # with open('../../data/tvb/v5.6_test_signer0_sim.pkl', 'rb') as f:
    #     test_s0_sim = pickle.load(f)

    # for i in range(len(dev)):
    #     name = dev[i]['name']
    #     dev[i]['text'] = name2text_dev[name]
    # for i in range(len(dev_s0)):
    #     name = dev_s0[i]['name']
    #     dev_s0[i]['text'] = name2text_dev[name]
    # for i in range(len(dev_sim)):
    #     name = dev_sim[i]['name']
    #     dev_sim[i]['text'] = Traditional2Simplified(name2text_dev[name])
    # for i in range(len(dev_s0_sim)):
    #     name = dev_s0_sim[i]['name']
    #     dev_s0_sim[i]['text'] = Traditional2Simplified(name2text_dev[name])
    
    # for i in range(len(test)):
    #     name = test[i]['name']
    #     test[i]['text'] = name2text_test[name]
    # for i in range(len(test_s0)):
    #     name = test_s0[i]['name']
    #     test_s0[i]['text'] = name2text_test[name]
    # for i in range(len(test_sim)):
    #     name = test_sim[i]['name']
    #     test_sim[i]['text'] = Traditional2Simplified(name2text_test[name])
    # for i in range(len(test_s0_sim)):
    #     name = test_s0_sim[i]['name']
    #     test_s0_sim[i]['text'] = Traditional2Simplified(name2text_test[name])

    # with open('../../data/tvb/v5.6_dev.pkl', 'wb') as f:
    #     pickle.dump(dev, f)
    # with open('../../data/tvb/v5.6_test.pkl', 'wb') as f:
    #     pickle.dump(test, f)
    # with open('../../data/tvb/v5.6_dev_signer0.pkl', 'wb') as f:
    #     pickle.dump(dev_s0, f)
    # with open('../../data/tvb/v5.6_test_signer0.pkl', 'wb') as f:
    #     pickle.dump(test_s0, f)
    # with open('../../data/tvb/v5.6_dev_sim.pkl', 'wb') as f:
    #     pickle.dump(dev_sim, f)
    # with open('../../data/tvb/v5.6_test_sim.pkl', 'wb') as f:
    #     pickle.dump(test_sim, f)
    # with open('../../data/tvb/v5.6_dev_signer0_sim.pkl', 'wb') as f:
    #     pickle.dump(dev_s0_sim, f)
    # with open('../../data/tvb/v5.6_test_signer0_sim.pkl', 'wb') as f:
    #     pickle.dump(test_s0_sim, f)

    #-------------------------------------------------single signer----------------------------------------------------
    # wer_dev = np.array([57.86, 42.24, 36.49, 35.05, 34.86])
    # wer_test = np.array([56.57, 42.09, 37.03, 35.22, 34.28])
    # slr_oov = np.array([4.27, 1.54, 0.7, 0.3, 0.0])

    # bleu_dev = np.array([10.72, 15.09, 18.14, 21.01, 21.06])
    # bleu_test = np.array([11.87, 16.25, 20.29, 22.79, 23.54])
    # slt_oov = np.array([1.35, 0.73, 0.37, 0.17, 0.17])

    # fig, (ax1,ax2) = plt.subplots(2,1)
    # x = np.arange(wer_dev.shape[0])
    # l1 = ax1.plot(x, wer_dev, color='#f3726d', marker='o', linestyle='dashed', label='Dev')
    # l2 = ax1.plot(x, wer_test, color='#60a9b7', marker='v', linestyle='dashed', label='Test')
    # ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ticks = ['25', '25', '50', '75', '100', '133']
    # ax1.set_xticklabels(ticks)
    # ax1.set_xlabel('Data Amount of Signer-1 (%)')
    # ax1.set_ylabel('WER (%)')
    # ax3 = ax1.twinx()
    # l3 = ax3.plot(x, slr_oov, color='grey', marker='X', linestyle='dashed', label='OOV')
    # ax3.set_ylabel('OOV (%)')
    # ax3.set_ylim(0,30)
    # ls = l1 + l2 + l3
    # labs = [l.get_label() for l in ls]
    # ax1.legend(ls, labs)

    # l4 = ax2.plot(x, bleu_dev, color='#f3726d', marker='o', linestyle='dashed', label='Dev')
    # l5 = ax2.plot(x, bleu_test, color='#60a9b7', marker='v', linestyle='dashed', label='Test')
    # # ax1.plot(x)
    # ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax2.set_xticklabels(ticks)
    # ax2.set_xlabel('Data Amount of Signer-1 (%)')
    # ax2.set_ylabel('BLEU-4')
    # ax4 = ax2.twinx()
    # l6 = ax4.plot(x, slt_oov, color='grey', marker='X', linestyle='dashed', label='OOV')
    # ax4.set_ylabel('OOV (%)')
    # ax4.set_ylim(0,30)
    # ls = l4 + l5 + l6
    # labs = [l.get_label() for l in ls]
    # ax2.legend(ls, labs)
    # plt.tight_layout()
    # plt.savefig('./single_signer.pdf')
    #-------------------------------------------------single signer----------------------------------------------------

    #-------------------------------------------------qual res----------------------------------------------------
    # def clean_tvb(s):
    #     op = []
    #     for t in s.split():
    #         if '<' in t and '>' in t:
    #             continue
    #         op.append(t)
    #     return ' '.join(op)
    
    # with open('../../data/results/tvb/rgb_s2g.pkl', 'rb') as f:
    #     s2g_rgb = pickle.load(f)
    # with open('../../data/results/tvb/pose_s2g.pkl', 'rb') as f:
    #     s2g_pose = pickle.load(f)
    # with open('../../data/results/tvb/two_s2g.pkl', 'rb') as f:
    #     s2g_two = pickle.load(f)

    # with open('../../data/results/tvb/rgb_s2t.pkl', 'rb') as f:
    #     s2t_rgb = pickle.load(f)
    # with open('../../data/results/tvb/pose_s2t.pkl', 'rb') as f:
    #     s2t_pose = pickle.load(f)
    # with open('../../data/results/tvb/two_s2t.pkl', 'rb') as f:
    #     s2t_two = pickle.load(f)

    # with open('../../data/tvb/dev.pkl', 'rb') as f:
    #     dev = pickle.load(f)

    # with open('../../data/tvb/v5.6_gloss2ids.pkl', 'rb') as f:
    #     gloss2ids = pickle.load(f)
    #     ids2gloss = {v:k for k,v in gloss2ids.items()}
    # with open('../../data/tvb/v5.6_gloss2ids_sim.pkl', 'rb') as f:
    #     gloss2ids_sim = pickle.load(f)
    # sim2trad = {}
    # for k,v in gloss2ids_sim.items():
    #     sim2trad[k] = ids2gloss[v]

    # with open('./tvb_qual_res_s2t.txt', 'w') as f:
    #     for item in dev:
    #         name = item['name']
    #         s2g_ref = clean_tvb(s2g_rgb[name]['gls_ref'])
    #         s2g_rgb_hyp = clean_tvb(s2g_rgb[name]['gls_hyp'])
    #         s2g_pose_hyp = clean_tvb(s2g_pose[name]['gls_hyp'])
    #         s2g_two_hyp = clean_tvb(s2g_two[name]['gls_hyp'])

    #         s2t_ref = Simplified2Traditional(s2t_rgb[name]['txt_ref'])
    #         s2t_rgb_hyp = Simplified2Traditional(s2t_rgb[name]['txt_hyp'])
    #         s2t_pose_hyp = Simplified2Traditional(s2t_pose[name]['txt_hyp'])
    #         s2t_two_hyp = Simplified2Traditional(s2t_two[name]['txt_hyp'])

    #         wer_rgb = wer_list(hypotheses=[s2g_rgb_hyp], references=[s2g_ref])['wer']
    #         wer_pose = wer_list(hypotheses=[s2g_pose_hyp], references=[s2g_ref])['wer']
    #         wer_two = wer_list(hypotheses=[s2g_two_hyp], references=[s2g_ref])['wer']

    #         bleu_rgb = bleu(references=[s2t_ref], hypotheses=[s2t_rgb_hyp], level='char')['bleu4']
    #         bleu_pose = bleu(references=[s2t_ref], hypotheses=[s2t_pose_hyp], level='char')['bleu4']
    #         bleu_two = bleu(references=[s2t_ref], hypotheses=[s2t_two_hyp], level='char')['bleu4']

    #         if (bleu_two > bleu_rgb and bleu_rgb > bleu_pose): #(wer_two < wer_rgb and wer_rgb < wer_pose):# or (bleu_two > bleu_rgb and bleu_rgb > bleu_pose):
    #             f.write(name+'\n')
    #             # f.write('GLS_REF: {}\n'.format(s2g_ref))
    #             # f.write('GLS_RGB_HYP_{:.2f}: {}\n'.format(wer_rgb, s2g_rgb_hyp))
    #             # f.write('GLS_POSE_HYP_{:.2f}: {}\n'.format(wer_pose, s2g_pose_hyp))
    #             # f.write('GLS_TWO_HYP_{:.2f}: {}\n\n'.format(wer_two, s2g_two_hyp))
                
    #             f.write('TXT_REF: {}\n'.format(item['text']))
    #             f.write('TXT_RGB_HYP_{:.2f}: {}\n'.format(bleu_rgb, s2t_rgb_hyp))
    #             f.write('TXT_POSE_HYP_{:.2f}: {}\n'.format(bleu_pose, s2t_pose_hyp))
    #             f.write('TXT_TWO_HYP_{:.2f}: {}\n'.format(bleu_two, s2t_two_hyp))
    #             f.write('\n')

    #-------------------------------------------------qual res----------------------------------------------------

    #------------------------------------------------v5.7 merge--------------------------------------------------dd
    # with open('../../data/tvb/v5.6_dev.pkl', 'rb') as f:
    #     dev = pickle.load(f)
    # with open('../../data/tvb/v5.6_test.pkl', 'rb') as f:
    #     test = pickle.load(f)
    # with open('../../data/tvb/v5.6_dev_signer0.pkl', 'rb') as f:
    #     dev_s0 = pickle.load(f)
    # with open('../../data/tvb/v5.6_test_signer0.pkl', 'rb') as f:
    #     test_s0 = pickle.load(f)
    # with open('../../data/tvb/v5.6_dev_sim.pkl', 'rb') as f:
    #     dev_sim = pickle.load(f)
    # with open('../../data/tvb/v5.6_test_sim.pkl', 'rb') as f:
    #     test_sim = pickle.load(f)
    # with open('../../data/tvb/v5.6_dev_signer0_sim.pkl', 'rb') as f:
    #     dev_s0_sim = pickle.load(f)
    # with open('../../data/tvb/v5.6_test_signer0_sim.pkl', 'rb') as f:
    #     test_s0_sim = pickle.load(f)
    # with open('../../data/tvb/v5.6_dev_remain.pkl', 'rb') as f:
    #     dev_r = pickle.load(f)
    # with open('../../data/tvb/v5.6_test_remain.pkl', 'rb') as f:
    #     test_r = pickle.load(f)

    # for item in dev_r:
    #     new_item = deepcopy(item)
    #     text = item['text']
    #     new_text = ''.join(text.strip().split())
    #     assert ' ' not in new_text
    #     new_text = Traditional2Simplified(new_text)
    #     new_gloss = Traditional2Simplified(item['gloss'])
    #     new_item['text'] = new_text
    #     new_item['gloss'] = new_gloss

    #     dev.append(item)
    #     dev_sim.append(new_item)
    #     if item['signer'] == 'Signer1':
    #         print(item['signer'])
    #         dev_s0.append(item)
    #         dev_s0_sim.append(new_item)
        
    # for item in test_r:
    #     new_item = deepcopy(item)
    #     text = item['text']
    #     new_text = ''.join(text.strip().split())
    #     assert ' ' not in new_text
    #     new_text = Traditional2Simplified(new_text)
    #     new_gloss = Traditional2Simplified(item['gloss'])
    #     new_item['text'] = new_text
    #     new_item['gloss'] = new_gloss

    #     test.append(item)
    #     test_sim.append(new_item)
    #     if item['signer'] == 'Signer1':
    #         print(item['signer'])
    #         test_s0.append(item)
    #         test_s0_sim.append(new_item)

    # with open('../../data/tvb/v5.7_dev.pkl', 'wb') as f:
    #     print(len(dev))
    #     pickle.dump(dev, f)
    # with open('../../data/tvb/v5.7_test.pkl', 'wb') as f:
    #     print(len(test))
    #     pickle.dump(test, f)
    # with open('../../data/tvb/v5.7_dev_signer0.pkl', 'wb') as f:
    #     print(len(dev_s0))
    #     pickle.dump(dev_s0, f)
    # with open('../../data/tvb/v5.7_test_signer0.pkl', 'wb') as f:
    #     print(len(test_s0))
    #     pickle.dump(test_s0, f)
    # with open('../../data/tvb/v5.7_dev_sim.pkl', 'wb') as f:
    #     print(len(dev_sim))
    #     pickle.dump(dev_sim, f)
    # with open('../../data/tvb/v5.7_test_sim.pkl', 'wb') as f:
    #     print(len(test_sim))
    #     pickle.dump(test_sim, f)
    # with open('../../data/tvb/v5.7_dev_signer0_sim.pkl', 'wb') as f:
    #     print(len(dev_s0_sim))
    #     pickle.dump(dev_s0_sim, f)
    # with open('../../data/tvb/v5.7_test_signer0_sim.pkl', 'wb') as f:
    #     print(len(test_s0_sim))
    #     pickle.dump(test_s0_sim, f)
    
    #------------------------------------------------v5.7 merge--------------------------------------------------
    # with open('../../data/tvb/TVB_results/twostream_signer0_100/dev/tvb_results.pkl', 'rb') as f:
    #     results = pickle.load(f)
    # gls_ref = [clean_tvb(results[n]['gls_ref']) for n in results]
    # gls_hyp = [clean_tvb(results[n]['gls_hyp']) for n in results]
    # wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
    # wer_results_per_sen = wer_list_per_sen(hypotheses=gls_hyp, references=gls_ref)
    # print(wer_results_per_sen['wer'])

    cc = OpenCC('s2t')
    with open('../SLG/data/T2G_results_dev.pkl', 'rb') as f:
        t2g = pickle.load(f)
    
    new_t2g = {}
    for k,v in t2g.items():
        gls_hyp = v['gls_hyp']
        gls_ref = v['gls_ref']
        # new_hyp = [Simplified2Traditional(g) for g in gls_hyp.split()]
        # new_hyp = ' '.join(new_hyp)
        # new_ref = [Simplified2Traditional(g) for g in gls_ref.split()]
        # new_ref = ' '.join(new_ref)
        new_hyp = cc.convert(gls_hyp)
        new_ref = cc.convert(gls_ref)
        new_t2g[k] = {'gls_hyp': new_hyp, 'gls_ref': new_ref}
    
    with open('../SLG/data/T2G_results_clean_cc_dev.pkl', 'wb') as f:
        pickle.dump(new_t2g, f)

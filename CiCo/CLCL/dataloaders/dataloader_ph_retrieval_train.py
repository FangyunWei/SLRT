from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch
import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataloaders.rawvideo_util import RawVideoExtractor
import random
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from textaugment import EDA

class ph_DataLoader_train(Dataset):
    """MSVD dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            feature_len=64,
            args=None
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.feature_len=feature_len
        self.subset = subset
        assert self.subset in ["train", "dev", "test"]
        sentance_id_path_dict = {}
        sentance_id_path_dict["train"] = os.path.join(self.data_path, "train.pkl")
        # sentance_id_path_dict["dev"] = os.path.join(self.data_path, "dev.txt")
        sentance_id_path_dict["test"] = os.path.join(self.data_path, "test.pkl")
        self.text_aug=args.text_aug
        self.features_path = os.path.join(self.features_path, self.subset)
        self.features_path_retrain=os.path.join(args.features_path_retrain, self.subset)
        self.combine_type=args.combine_type
        with open(sentance_id_path_dict[self.subset], 'rb') as f:
            captions = pickle.load(f)
        self.captions=captions
        self.emd = EDA()
        #video_ids: video_id
        #video_dict: video_path
        #

        sentance_ids = captions.keys()
        sentences_dict = {}
        for sentance_id in sentance_ids:
            text=captions[sentance_id]['text']
            sentences_dict[len(sentences_dict)] = (sentance_id, text)
        self.sentences_dict = sentences_dict

        self.sample_len = 0
        self.video_dict = {}
        self.video_dict_retrain={}
        self.cut_off_points = []
        video_num=0
        self.alpha=args.alpha

        for sentance_id in sentance_ids:

            video_path=os.path.join(self.features_path,captions[sentance_id]['video_name']+'.pkl')
            video_path_retrain=os.path.join(self.features_path,captions[sentance_id]['video_name']+'.pkl')

            if sentance_id not in self.video_dict:
                self.video_dict[sentance_id]=[video_path]
                self.video_dict_retrain[sentance_id]=[video_path_retrain]
            else:
                self.video_dict[sentance_id].append(video_path)
                self.video_dict_retrain[sentance_id].append(video_path_retrain)

            video_num+=len(captions[sentance_id])
            self.cut_off_points.append(video_num)

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(self.video_dict)
            assert len(self.cut_off_points) == self.sentence_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Sentance number: {}".format(len(self.sentences_dict )))
        print("Total Paire: {}".format(video_num))

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, ids):
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        pairs_text_aug = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask_aug = np.zeros((k, self.max_words), dtype=np.long)

        choice_sentance_ids,text=self.sentences_dict[ids][0],self.sentences_dict[ids][1]
        choice_sentance_ids=[choice_sentance_ids]

        for i, sentance_id in enumerate(choice_sentance_ids):
            is_aug=0
            choose_porb=0.5

            if self.text_aug==True and random.random()>1-choose_porb:
                is_aug=1
                emd_aug=self.emd.random_swap
                text_aug = emd_aug(text)
                if isinstance(text_aug, list):
                    text_aug = ' '.join(text_aug)
                words_aug = self.tokenizer.tokenize(text_aug)
                words_aug = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words_aug
                total_length_with_CLS = self.max_words - 1
                words_index = [0]
                if len(words_aug) > total_length_with_CLS:
                    # selected_index = list(np.arange(len(words)-1))
                    all_index = list(np.linspace(1, len(words_aug) - 1, total_length_with_CLS - 1, dtype=int))
                    # all_index = list(np.random.randint(1, len(words) - 1, total_length_with_CLS - 1, dtype=int))
                    selected_index = sorted(all_index)
                    words_index += selected_index
                    words_aug = list(np.array(words_aug)[words_index])
                    # words = words[:total_length_with_CLS]
                words_aug = words_aug + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

                input_ids_aug = self.tokenizer.convert_tokens_to_ids(words_aug)
                input_mask_aug = [1] * len(input_ids_aug)
                while len(input_ids_aug) < self.max_words:
                    input_ids_aug.append(0)
                    input_mask_aug.append(0)

                pairs_text_aug[i] = np.array(input_ids_aug)
                pairs_mask_aug[i] = np.array(input_mask_aug)


            words = self.tokenizer.tokenize(text)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            words_index=[0]
            if len(words) > total_length_with_CLS:
                #selected_index = list(np.arange(len(words)-1))
                all_index = list(np.linspace(1, len(words) - 1, total_length_with_CLS-1, dtype=int))
                # all_index = list(np.random.randint(1, len(words) - 1, total_length_with_CLS - 1, dtype=int))
                selected_index=sorted(all_index)
                words_index+=selected_index
                words=list(np.array(words)[words_index])
                # words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            if is_aug == 0:
                pairs_text_aug = pairs_text
                pairs_mask_aug = pairs_mask

        return pairs_text, pairs_mask, pairs_segment, choice_sentance_ids[0],pairs_text_aug,pairs_mask_aug

    def _get_rawvideo(self, sentance_id):
        videos=self.video_dict[sentance_id]
        feature_len=self.feature_len
        rands = random.randint(0, len(videos) - 1)
        video_file_path=videos[rands]


        videos_retrain=self.video_dict_retrain[sentance_id]
        video_retrain_path=videos_retrain[rands]

        if self.combine_type!="cat":
            video_feature = torch.zeros((1024, feature_len, 1))
            video_mask = np.ones(feature_len + 1, dtype=np.long)
            video_mask[0] = 1
            with open(video_retrain_path, 'rb') as f:
                item = pickle.load(f)
                video_feature_pre = item['feature']

            if self.combine_type=='sum':
                with open(video_file_path, 'rb') as f:
                    item = pickle.load(f)
                    video_feature_pre = (1-self.alpha)*video_feature_pre+self.alpha*item['feature']

            video_feature_pre=torch.Tensor(video_feature_pre).transpose(0, 1)
            video_feature_pre=video_feature_pre.view(video_feature_pre.shape[0], -1, 1)

            video_len=video_feature_pre.shape[1]


            if video_len>=feature_len:
                all_index=list(np.linspace(0, video_len-1, feature_len, dtype=int))
                choosen_idx = sorted(all_index)
            else:
                choosen_idx=range(video_len)
                choosen_idx=list(choosen_idx)
            for i in range(len(choosen_idx)):
                video_feature[:,i,:]=video_feature_pre[:,choosen_idx[i],:]
                video_mask[i+1]=0
        elif self.combine_type=="cat":
            video_feature = torch.zeros((1024, feature_len,2, 1))
            video_mask = np.ones((feature_len + 1), dtype=np.long)
            video_mask[0] = 1
            with open(video_retrain_path, 'rb') as f:
                item = pickle.load(f)
                video_feature_pre = item['feature']
            with open(video_file_path, 'rb') as f:
                item = pickle.load(f)
                video_feature_pre_ori = item['feature']

            video_feature_pre=torch.Tensor(video_feature_pre).transpose(0, 1)
            video_feature_pre=video_feature_pre.view(video_feature_pre.shape[0], -1, 1)
            video_feature_pre_ori=torch.Tensor(video_feature_pre_ori).transpose(0, 1)
            video_feature_pre_ori=video_feature_pre_ori.view(video_feature_pre.shape[0], -1, 1)


            video_len=video_feature_pre.shape[1]
            if video_len>=feature_len:
                all_index=list(np.linspace(0, video_len-1, 64, dtype=int))
                choosen_idx = sorted(all_index)
            else:
                choosen_idx=range(video_len)
                choosen_idx=list(choosen_idx)
            for i in range(len(choosen_idx)):
                video_feature[:,i,0,:]=video_feature_pre[:,choosen_idx[i],:]
                video_feature[:,i,1,:]=video_feature_pre_ori[:,choosen_idx[i],:]

                video_mask[i+1]=0
            video_feature=video_feature.view(1024, feature_len*2, 1)
            video_mask=video_mask.reshape(-1)


        return video_feature, video_mask

    def __getitem__(self, idx):
        pairs_text, pairs_mask, pairs_segment, choice_sentance_ids,pairs_text_aug,pairs_mask_aug = self._get_text(idx)

        video_feature, video_mask = self._get_rawvideo(choice_sentance_ids)
        return pairs_text, pairs_mask, pairs_segment, video_feature, video_mask,pairs_text_aug,pairs_mask_aug

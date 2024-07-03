import gzip,json,pickle,os
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict
dataset2result_file = {
    'csl':'data/results/csl-daily_s2g/gloss_logits/{}.pkl',
    'phoenix': 'data/results/phoenix-2014t_s2g/gloss_logits/{}.pkl',
    'phoenix2014': 'data/results/phoenix-2014_s2g/gloss_logits/{}.pkl'
}
dataset2gloss2id_file = {
    'csl': '../../data/csl-daily/gloss2ids.pkl',
    'phoenix': '../../data/phoenix_2014t/gloss2ids.pkl',
    'phoenix2014': '../../data/phoenix2014/gloss2ids.pkl'
}
dataset2data_file = {
    'csl': '../../data/csl-daily/csl.{}',
    'phoenix': '../../data/phoenix_2014t/phoenix14t.{}',
    'phoenix2014': '../../data/phoenix2014/phoenix2014.{}'
}
from dataset.VideoLoader import read_img
dataset2zip_file = {
    'csl':'../../data/csl-daily/sentence_frames-512x512.zip',
    'phoenix':'../data/PHOENIX/PHOENIX2014T_videos.zip',
    'phoenix2014': '../../data/phoenix-2014/phoenix-2014-videos.zip'
}
datasetname2format = {
    'csl': '{}@sentence_frames-512x512/{}/{:06d}.jpg',
    'phoenix': '{}@images/{}/images{:04d}.png',
    'phoenix2014': '{}@{}.avi_pid0_fn{:06d}-0.png'
}


def logits2segment(gloss_probabilities, gloss_labels, num_frames, only_blank=False):
    P = gloss_probabilities
    K = gloss_probabilities.shape[0]
    U = len(gloss_labels)
    ref_id = gloss_labels
    P_ = P[:,ref_id]
    U = len(ref_id)
    U_ = 2*U+1
    ref_id_ = [ref_id[i//2] if i%2==1 else 0 for i in range(U_)]
    ref_id_rank = [i//2+1 if i%2==1 else 0 for i in range(U_)]
    #forward
    alpha = torch.zeros(K, U_)
    alpha[0,0], alpha[0,1] = P[0,0], P[0, ref_id_[1]]
    path = torch.zeros(K, U_)

    ranks_wo_blank = torch.argsort(gloss_probabilities[:,1:]*-1, dim=1)+1
    #alpha[0,>=2] = 0
    for k in range(1, K):
        for u in range(0,U_):
            cand_list = [alpha[k-1, u].item()]  
            inds = [u]
            if u-1>=0:
                cand_list.append(alpha[k-1, u-1].item())
                inds.append(u-1)
            if u-2>=0 and ref_id_[u]!=0 and ref_id_[u-2]!=ref_id_[u]:
                cand_list.append(alpha[k-1, u-2].item())
                inds.append(u-2)
            alpha[k,u] = P[k, ref_id_[u]]*max(cand_list)
            path[k,u] = inds[np.argmax(cand_list)]
            #print(k,u, path[k,u].item())
    path = path.long()
    #backward
    if alpha[K-1,U_-1]>alpha[K-1,U_-2]:
        final_path = [U_-1]
    else:
        final_path = [U_-2]
    for k in range(K-1,0,-1):
        final_path = [path[k,final_path[0]].item()]+final_path
    final_path2ids = [ref_id_rank[int(p)] for p in final_path]

    #print(final_path2ids)
    #print(ranks_wo_blank[:,0])

    s = 0
    segments = []
    if not only_blank:
        for i in range(1, U+1):
            while s<K and final_path2ids[s]!=i:
                s+= 1
            left = min(s,K-1) #if s==K ->K-1
    #         if i==1:
    #             left = 0
    #         else:
            while left>=0 and ranks_wo_blank[left,0]==ref_id[i-1] and (final_path2ids[left]!=i-1 or final_path2ids[left]==0):
                left -= 1
            if left!=K-1 and left!=s:
                left += 1
            while s<K and final_path2ids[s]==i:
                s+= 1
            right = s
    #         if i==U:
    #             right = K
    #         else:
            while right<K and ranks_wo_blank[right,0]==ref_id[i-1] and final_path2ids[right]!=i+1:
                right += 1
            if right!=0:
                right -=1 #make it inclusive
            #print('i=',i,'right=',right)
            #print(i, left, right)
            if right>left:
                s_, e_ = left*4, right*4 #inclusive
            elif right==left:
                if left==0:
                    s_, e_ = 0, 4 # the first 5 frames
                elif left==gloss_probabilities.shape[0]-1:
                    s_, e_ = left*4-2, num_frames-1
                else:
                    s_, e_ = left*4-2, right*4+2 #5 frames
            e_ = min(e_, num_frames-1)
            s_ = max(s_, 0)
            segments.append([ref_id[i-1],[s_,e_]])
    
    else:
        # crop blank, final_path2ids=0
        left = right = 0
        max_len = len(final_path2ids)
        while right < max_len:
            while right < max_len and final_path2ids[right] != 0:
                right += 1
            if right == max_len:
                break
            while right < max_len and final_path2ids[right] == 0:
                right += 1

            left = right - 1
            if final_path2ids[left] != 0:
                break

            while left > -1 and final_path2ids[left] == 0:
                left -= 1
            left += 1

            if right >= left+2:  #at least 2 time steps
                s = max(left*4, 0)
                e = min((right-1)*4, num_frames-1)
                segments.append([0, [s,e]])

    return segments


if __name__ == '__main__':
    datasetname = 'phoenix'
    outputdir = '../../data/phoenix_2014t/'
    write = True
    only_blank = True
    result_file = dataset2result_file[datasetname]
    os.makedirs(outputdir, exist_ok=True)
    vocab2fre = {} #for training
    with open(dataset2gloss2id_file[datasetname],'rb') as f:
        gloss2ids = pickle.load(f)
    id2gloss = {i:g for g, i in gloss2ids.items()}
    for split in ['dev', 'test']:
        with gzip.open(dataset2data_file[datasetname].format(split),'rb') as f:
            data = pickle.load(f)
        with gzip.open(result_file.format(split),'rb') as f:
            gloss_logits = pickle.load(f)
        name2logits = {g['name']:g['sign'] for g in gloss_logits}
        islr_data = []
        for ex in tqdm(data):
            gloss_prob = torch.softmax(name2logits[ex['name']],dim=-1)
            if datasetname=='phoenix':
                gloss_labels = [gloss2ids[g.lower()] for g in ex['gloss'].split(' ')] 
            else:
                gloss_labels = []
                for g in ex['gloss'].split(' '):
                    if len(g) and g in gloss2ids:
                        gloss_labels.append(gloss2ids[g])
                    elif not g in gloss2ids:
                        gloss_labels.append(gloss2ids['<unk>'])
            segments = logits2segment(gloss_prob, gloss_labels, ex['num_frames'], only_blank)
            # print(segments)
            name_base = ex['name']
            for gid, (s,e) in segments: #inclusive
                if not only_blank:
                    gls = id2gloss[gid]
                else:
                    gls = '<blank>'
                e += 1 #inclusive -> exclusive
                if e>s:
                    name = f'{gls}_{name_base}_[{s}:{e}]'
                    #print(name)
                    islr_data.append(
                        {'video_file':name_base, 'name':name, 'label':gls, 'seq_len':e-s, 'start':s, 'end':e}
                    )
                #input()
        seq_lens = [t['seq_len'] for t in islr_data]
        print(split, len(islr_data), np.mean(seq_lens), np.std(seq_lens))
        if write:
            if not only_blank:
                with open(f'{outputdir}/{datasetname}_iso.{split}','wb') as f:
                    pickle.dump(islr_data, f)
            else:
                with open(f'{outputdir}/{datasetname}_iso_blank.{split}','wb') as f:
                    pickle.dump(islr_data, f)

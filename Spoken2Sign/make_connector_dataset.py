import gzip, pickle, torch, os
from tqdm import tqdm
from collections import defaultdict


if __name__ == '__main__':
    dataset = 'phoenix'
    with open('../../data/phoenix/keypoints_3d_mesh.pkl', 'rb') as f:
        kps = pickle.load(f)
    for split in ['train', 'dev', 'test']:
        # with open('../../data/phoenix/train.pkl', 'rb') as f:
        #     ann = pickle.load(f)
        # with open('../../data/phoenix/phoenix_iso.{}'.format(split), 'rb') as f:
        #     iso_ann = pickle.load(f)
        
        # vfile2clips = defaultdict(list)
        # for item in iso_ann:
        #     vfile2clips[item['video_file']].append(item)
        # for k,v in vfile2clips.items():
        #     sorted_v = sorted(v, key=lambda x:x['start'])
        #     vfile2clips[k] = sorted_v
        
        # # pairs
        # pairs = []
        # for vfile, clips in vfile2clips.items():
        #     for idx in range(len(clips)-1):
        #         c1, c2 = clips[idx], clips[idx+1]
        #         if c1['end'] >= c2['start'] or c2['start']-c1['end']>9:
        #             continue
        #         pairs.append((c1,c2))
        # print('len of pairs:', len(pairs))
        # with open('../../data/phoenix/iso_clip_pairs.{}'.format(split), 'wb') as f:
        #     pickle.dump(pairs, f)


        with open('../../data/phoenix/iso_clip_pairs.{}'.format(split), 'rb') as f:
            pairs = pickle.load(f)
        count = 0
        clean_pairs = []
        for p in pairs:
            item1, item2 = p
            st1, end1 = item1['start'], item1['end']
            st2, end2 = item2['start'], item2['end']
            kp = kps[item1['video_file']]['keypoints_3d']
            if end2 > kp.shape[0]:
                count += 1
                continue
            clean_pairs.append(p)
    
        print(count, len(pairs))
        with open('../../data/phoenix/iso_clip_pairs.{}'.format(split), 'wb') as f:
            print(len(clean_pairs))
            pickle.dump(clean_pairs, f)

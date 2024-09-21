import pickle
import gzip
import json
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy


if __name__ == '__main__':
    split = 'train'
    with open('../../data/phoenix_2014t/phoenix_iso.{}'.format(split), 'rb') as f:
        data = pickle.load(f)
    with open('../../data/phoenix_2014t/phoenix_iso_blank.{}'.format(split), 'rb') as f:
        data1 = pickle.load(f)
    data = [*data, *data1]
    with gzip.open('../../data/phoenix_2014t/phoenix.{}'.format(split), 'rb') as f:
        ori = pickle.load(f)
    
    vfile2items = defaultdict(list)
    for item in data:
        vfile2items[item['video_file']].append(item)
    vfile2len = {}
    for item in ori:
        vfile2len[item['name']] = item['num_frames']
    
    bag2items = defaultdict(list)
    bag_idx = 0
    win_size = 16
    for vfile, item_lst in tqdm(vfile2items.items()):
        vlen = vfile2len[vfile]
        for item in item_lst:
            start, end = item['start'], item['end']
            base_start, base_end = start, end
            new_item = deepcopy(item)
            new_item['bag'] = bag_idx
            new_item['aug'] = 0
            new_item['base_start'], new_item['base_end'] = base_start, base_end
            bag2items[str(bag_idx)].append(new_item)
            for cen in range(start, end):
                new_start = cen-win_size//2
                new_end = new_start+win_size
                new_start = max(0, new_start)
                new_end = min(vlen, new_end)
                new_item = {'video_file': vfile, 'name': '{}_{}_[{}:{}]'.format(item['label'],vfile,new_start,new_end), 'label': item['label'], \
                            'seq_len': new_end-new_start, 'start': new_start, 'end': new_end, 'bag': bag_idx, 'aug': 1, 'base_start': base_start, 'base_end': base_end}
                bag2items[str(bag_idx)].append(new_item)
            bag_idx += 1

    with open('../../data/phoenix_2014t/phoenix_iso_center_label_bag2items.{}'.format(split), 'wb') as f:
        pickle.dump(bag2items, f)
    
    if split == 'train':
        # create vocab file
        vocab = ['<blank>']
        for k,v in bag2items.items():
            for item in v:
                if item['label'] not in vocab:
                    vocab.append(item['label'])
        with open('../../data/phoenix_2014t/phoenix_iso_with_blank.vocab', 'w') as f:
            json.dump(vocab, f)

    if split in ['dev', 'test']:
        new_data = []
        for k,v in bag2items.items():
            new_data.extend(v)
        with open('../../data/phoenix_2014t/phoenix_iso_center_label.{}'.format(split), 'wb') as f:
            pickle.dump(new_data, f)
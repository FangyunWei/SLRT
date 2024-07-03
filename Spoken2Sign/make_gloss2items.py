import pickle, gzip, json
from collections import defaultdict


if __name__ == '__main__':
    with open('../../data/phoenix/phoenix_islr/prediction/train/name_prob.pkl', 'rb') as f:
        name_prob = pickle.load(f)
    with open('../../data/phoenix/phoenix_iso_with_blank.vocab', 'r') as f:
        vocab = json.load(f)
    with open('../../data/phoenix/phoenix_iso.train', 'rb') as f:
        train = pickle.load(f)
    
    gloss2items = defaultdict(list)
    for item in train:
        gloss = item['label']
        name = item['name']
        gls_index = vocab.index(gloss)
        score = name_prob[name][gls_index].item()
        gloss2items[gloss].append((item, score))
    
    for k,v in gloss2items.items():
        s_v = sorted(v, key=lambda x:-x[1])
        gloss2items[k] = s_v
    
    with open('data/gloss2items.pkl', 'wb') as f:
        pickle.dump(gloss2items, f)


import torch, os, gzip, pickle, json, numpy as np
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prune word-embedding layer and compute gloss embeddings")
    parser.add_argument(
        "--input_dir",
        default="pretrained_models/mBart",
        type=str,
    )
    args = parser.parse_args()

    tokenizer = MBartTokenizer.from_pretrained(args.input_dir)
    model = MBartForConditionalGeneration.from_pretrained(args.input_dir)
    print('Vocab size = ', tokenizer.vocab_size)

    full_embedding_weight = model.model.shared.weight
    full_final_logits_bias = model.final_logits_bias

    with open(os.path.join(os.path.join(args.input_dir, 'config.json')),'r') as f:
        config_json = json.load(f)

    # 1. Prune embedding layer
    new_embedding_weight_list = []
    new_final_logits_bias_list = []
    map_ids = {}
    ## 1.1 special tokens
    with open(os.path.join(args.input_dir, 'tokenizer.json'),'r') as f:
        tokenizer_json = json.load(f)
    print('Special tokens #', len(tokenizer_json['added_tokens']))
    
    for new_id, added_token in enumerate(tokenizer_json['added_tokens']):
        id_ = added_token['id']
        new_embedding_weight_list.append(full_embedding_weight[id_,:])
        new_final_logits_bias_list.append(full_final_logits_bias[:,id_])
        map_ids[id_] = new_id
        print('{} {}->{}'.format(added_token['content'], id_, new_id))
from json import decoder
import torch, pickle, json
from collections import defaultdict
from transformers import MBartTokenizer

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, ignore_index: int=-100):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    for ii,ind in enumerate(index_of_eos.squeeze(-1)):
        input_ids[ii, ind:] = ignore_index
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

class BaseTokenizer(object):
    def __init__(self, tokenizer_cfg):
        self.tokenizer_cfg = tokenizer_cfg
    def __call__(self, input_str):
        pass


class TextTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)

        self.level = tokenizer_cfg.get('level','sentencepiece')
        if self.level == 'word':
            self.min_freq = tokenizer_cfg.get('min_freq',0)
            with open(tokenizer_cfg['tokenizer_file'],'r') as f:
                tokenizer_info = json.load(f)
            self.word2fre, self.special_tokens = tokenizer_info['word2fre'], tokenizer_info['special_tokens']
            self.id2token = self.special_tokens[:]
            for w in sorted(self.word2fre.keys(), key=lambda w: self.word2fre[w])[::-1]:
                f = self.word2fre[w]
                if f>=self.min_freq:
                    self.id2token.append(w)
            self.token2id = {t: id_ for id_, t in enumerate(self.id2token)}
            self.pad_index, self.eos_index, self.unk_index, self.sos_index = \
                self.token2id['<pad>'], self.token2id['</s>'], self.token2id['<unk>'], self.token2id['<s>']
            self.token2id = defaultdict(lambda:self.unk_index, self.token2id)
            self.ignore_index = self.pad_index
        elif self.level == 'sentencepiece':
            self.tokenizer = MBartTokenizer.from_pretrained(
                **tokenizer_cfg) #tgt_lang
            self.pad_index = self.tokenizer.convert_tokens_to_ids('<pad>')
            self.ignore_index = self.pad_index
            
            self.pruneids_file = tokenizer_cfg['pruneids_file']
            with open(self.pruneids_file, 'rb') as f:
                self.pruneids = pickle.load(f) # map old2new #gls2token
                for t in ['<pad>','<s>','</s>','<unk>']:
                    id_ = self.tokenizer.convert_tokens_to_ids(t)
                    assert self.pruneids[id_] == id_, '{}->{}'.format(id_, self.pruneids[id_])
            self.pruneids_reverse = {i2:i1 for i1,i2 in self.pruneids.items()}
            self.lang_index = self.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]
            self.sos_index = self.lang_index
            self.eos_index = self.pruneids[self.tokenizer.convert_tokens_to_ids('</s>')]
        else:
            raise ValueError

    def generate_decoder_labels(self, input_ids):
        decoder_labels = torch.where(
            input_ids==self.lang_index,  #already be mapped into pruned_vocab
            torch.ones_like(input_ids)*self.ignore_index, input_ids)
        return decoder_labels

    def generate_decoder_inputs(self, input_ids):
        decoder_inputs = shift_tokens_right(input_ids, 
            pad_token_id=self.pad_index,
            ignore_index=self.pad_index)
        return decoder_inputs

    def prune(self, input_ids):
        pruned_input_ids = []
        for  single_seq in input_ids:
            pruned_single_seq = []
            for id_ in single_seq:
                if not id_ in self.pruneids:
                    new_id = self.pruneids[self.tokenizer.convert_tokens_to_ids('<unk>')]
                    print(id_)
                    print(self.tokenizer.convert_ids_to_tokens(id_))
                else:
                    new_id = self.pruneids[id_]
                pruned_single_seq.append(new_id)
            pruned_input_ids.append(pruned_single_seq)
        return torch.tensor(pruned_input_ids, dtype=torch.long)
    
    def prune_reverse(self, pruned_input_ids):
        batch_size, max_len = pruned_input_ids.shape
        input_ids = pruned_input_ids.clone()
        for b in range(batch_size):
            for i in range(max_len):
                id_ = input_ids[b,i].item()
                if not id_ in self.pruneids_reverse:
                    new_id = self.tokenizer.convert_tokens_to_ids('<unk>')
                else:
                    new_id = self.pruneids_reverse[id_]
                input_ids[b,i] = new_id
        return input_ids
    
    def __call__(self, input_str):
        if self.level == 'sentencepiece':
            with self.tokenizer.as_target_tokenizer():
                raw_outputs = self.tokenizer(input_str, 
                    #return_tensors="pt", 
                    return_attention_mask=True,
                    return_length=True,
                    padding='longest')
            outputs = {}
            pruned_input_ids = self.prune(raw_outputs['input_ids'])
            outputs['labels'] = self.generate_decoder_labels(pruned_input_ids)
            outputs['decoder_input_ids'] = self.generate_decoder_inputs(pruned_input_ids)
        elif self.level == 'word':
            #input as a batch
            batch_labels, batch_decoder_input_ids, batch_lengths = [],[],[]
            for text in input_str:
                labels, decoder_input_ids = [], [self.sos_index]
                for ti, t in enumerate(text.split()):
                    id_ = self.token2id[t]
                    labels.append(id_)
                    decoder_input_ids.append(id_)
                labels.append(self.eos_index)
                batch_labels.append(labels)
                batch_decoder_input_ids.append(decoder_input_ids)
                batch_lengths.append(len(labels))
            #padding
            max_length = max(batch_lengths)
            padded_batch_labels, padded_batch_decoder_input_ids = [], []
            for labels, decoder_input_ids in zip(batch_labels, batch_decoder_input_ids):
                padded_labels = labels + [self.pad_index]*(max_length-len(labels))
                padded_decoder_input_ids = decoder_input_ids + [self.ignore_index]*(max_length-len(decoder_input_ids))
                assert len(padded_labels)==len(padded_decoder_input_ids)
                padded_batch_labels.append(padded_labels)
                padded_batch_decoder_input_ids.append(padded_decoder_input_ids)
            outputs = {
                'labels': torch.tensor(padded_batch_labels, dtype=torch.long),
                'decoder_input_ids': torch.tensor(padded_batch_decoder_input_ids, dtype=torch.long)
            }
        else:
            raise ValueError
        return outputs 
    
    def batch_decode(self, sequences):
        #remove the first token (bos)
        sequences = sequences[:,1:]
        if self.level == 'sentencepiece':
            sequences_ = self.prune_reverse(sequences)
            decoded_sequences = self.tokenizer.batch_decode(sequences_, skip_special_tokens=True)
            if 'de' in self.tokenizer.tgt_lang:
                for di, d in enumerate(decoded_sequences):
                    if len(d)>2 and d[-1]=='.' and d[-2]!=' ':
                        d = d[:-1]+ ' .'
                        decoded_sequences[di] = d 
        elif self.level == 'word':
            #... .</s>
            decoded_sequences = [' '.join([self.id2token[s] for s in seq]) for seq in sequences]
        else:
            raise ValueError
        return decoded_sequences
class BaseGlossTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        with open(tokenizer_cfg['gloss2id_file'],'rb') as f:
            self.gloss2id = pickle.load(f) #
        self.gloss2id = defaultdict(lambda: self.gloss2id['<unk>'], self.gloss2id)
        #check 
        ids = [id_ for gls, id_ in self.gloss2id.items()]
        assert len(ids)==len(set(ids))
        self.id2gloss = {}
        for gls, id_ in self.gloss2id.items():
            self.id2gloss[id_] = gls        
        self.lower_case = tokenizer_cfg.get('lower_case',True)
        
    def convert_tokens_to_ids(self, tokens):
        if type(tokens)==list:
            return [self.convert_tokens_to_ids(t) for t in tokens]
        else:
            return self.gloss2id[tokens]

    def convert_ids_to_tokens(self, ids):
        if type(ids)==list:
            return [self.convert_ids_to_tokens(i) for i in ids]
        else:
            return self.id2gloss[ids]
    
    def __len__(self):
        return len(self.id2gloss)


class GlossTokenizer_S2G(BaseGlossTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        if '<s>' in self.gloss2id:
            self.silence_token = '<s>'
            self.silence_id = self.convert_tokens_to_ids(self.silence_token)
        elif '<si>' in self.gloss2id:
            self.silence_token = '<si>'
            self.silence_id = self.convert_tokens_to_ids(self.silence_token)
        else:
            raise ValueError            
        assert self.silence_id==0, (self.silence_id)
        self.pad_token = '<pad>'
        self.pad_id = self.convert_tokens_to_ids(self.pad_token)

        if 'dataset2dic' in tokenizer_cfg:
            with open(tokenizer_cfg['dataset2dic'],'rb') as f:
                self.dataset2dic = pickle.load(f)
            self.dataset2ids = {}
            self.dataset2id_inv = {}
            for datasetname in self.dataset2dic:
                self.dataset2ids[datasetname] = \
                    sorted([id_ for gls,id_ in self.dataset2dic[datasetname].items()])
                self.dataset2id_inv[datasetname] = \
                    {old_i:new_i for new_i, old_i in enumerate(self.dataset2ids[datasetname])} #convert label
                assert len(set(self.dataset2ids[datasetname]))==len(self.dataset2ids[datasetname]), datasetname
                self.dataset2dic[datasetname] = defaultdict(lambda: self.dataset2dic[datasetname]['<unk>'], self.dataset2dic[datasetname])
        else:
            self.dataset2dic = defaultdict(lambda: self.gloss2id)
            ids = sorted([id_ for gls,id_ in self.gloss2id.items()])
            self.dataset2ids = defaultdict(lambda: ids)
            self.dataset2id_inv = defaultdict(lambda: {old_i:new_i for new_i, old_i in enumerate(ids)})
    
 
    def __call__(self, batch_gls_seq, datasetname, pretokenized=False):
        if pretokenized:
            max_length = max([len(gls_seq) for gls_seq in batch_gls_seq])
        else:
            max_length = max([len(gls_seq.split()) for gls_seq in batch_gls_seq])
        gls_lengths, batch_gls_ids = [], []
        for ii, gls_seq in enumerate(batch_gls_seq):
            #gls_ids = [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in gls_seq.split()]
            if pretokenized:
                gls_ids = gls_seq
                gls_lengths.append(len(gls_ids))
                gls_ids = gls_ids + (max_length-len(gls_ids))*[self.pad_id] #already tokenized in local_id
            else:
                gls_ids = [self.dataset2dic[datasetname][gls.lower() if self.lower_case else gls] \
                    for gls in gls_seq.split()] #-> to global vocab
                gls_lengths.append(len(gls_ids))
                gls_ids = gls_ids+(max_length-len(gls_ids))*[self.pad_id]
                gls_ids = [self.dataset2id_inv[datasetname][i] for i in gls_ids] #to local vocab
            batch_gls_ids.append(gls_ids)
        gls_lengths = torch.tensor(gls_lengths)
        batch_gls_ids = torch.tensor(batch_gls_ids)
        return {'gls_lengths':gls_lengths, 'gloss_labels': batch_gls_ids}

    def convert_ids_to_tokens(self, ids, datasetname):
        if type(ids)==list:
            return [self.convert_ids_to_tokens(i, datasetname) for i in ids]
        else:
            return self.id2gloss[self.dataset2ids[datasetname][ids]]

class GlossTokenizer_G2T(BaseGlossTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        self.src_lang = tokenizer_cfg['src_lang']
    def __call__(self, batch_gls_seq):
        #batch
        max_length = max([len(gls_seq.split()) for gls_seq in batch_gls_seq])+2 #include </s> <lang>
        batch_gls_ids = []
        attention_mask = torch.zeros([len(batch_gls_seq), max_length], dtype=torch.long)
        for ii, gls_seq in enumerate(batch_gls_seq):
            gls_ids = [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in gls_seq.split()]
            #add </s> <lang> + padding
            gls_ids = gls_ids + [self.gloss2id['</s>'],self.gloss2id[self.src_lang]]
            attention_mask[ii,:len(gls_ids)] = 1
            gls_ids = gls_ids + (max_length-len(gls_ids))*[self.gloss2id['<pad>']]
            batch_gls_ids.append(gls_ids)
        input_ids = torch.tensor(batch_gls_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {'input_ids':input_ids, 'attention_mask':attention_mask}
            


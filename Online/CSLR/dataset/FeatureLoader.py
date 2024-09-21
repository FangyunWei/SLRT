import torch


def load_batch_feature(features):
    #features list of tensor
    batch_features, lengths = [], []
    lengths = [f.shape[0] for f in features]
    max_length = max(lengths)
    mask = torch.zeros([len(features), max_length],dtype=torch.long)
    for ii,f in enumerate(features):
        if f.shape[0]<max_length:
            pad_len = max_length-f.shape[0]
            padding = torch.zeros([pad_len, f.shape[1]], dtype=f.dtype, device=f.device)
            padded_feature = torch.cat([f, padding], dim=0)
            batch_features.append(padded_feature)
        else:
            batch_features.append(f)
        mask[ii,:f.shape[0]] =1
    batch_features = torch.stack(batch_features, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return batch_features, mask, lengths

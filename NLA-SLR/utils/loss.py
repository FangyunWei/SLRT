from multiprocessing.sharedctypes import Value
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math


class LabelSmoothCE(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100, word_emb_tab=None, norm_type='softmax', temp=1.0, variant=None):
        super(LabelSmoothCE, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.word_emb_sim = None
        if word_emb_tab is not None:
            self.word_emb_sim = t.matmul(F.normalize(word_emb_tab, dim=-1), F.normalize(word_emb_tab, dim=-1).T)
            self.norm_type = norm_type
            self.temp = temp
        self.variant = variant


    def forward(self, logits, label, topk_idx=None, mixup_lam=None, y_a=None, y_b=None, **kwargs):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with t.no_grad():
            # print(self.variant)
            if self.variant is None:
                num_classes = logits.size(1)
                label = label.clone().detach()
                ignore = label.eq(self.lb_ignore)
                n_valid = ignore.eq(0).sum()
                label[ignore] = 0
                lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / (num_classes-1)
                lb_one_hot = t.empty_like(logits).fill_(
                    lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

            elif 'dual' in self.variant:
                # assert topk_idx is not None
                B,K,N = logits.shape
                label = label.clone().detach()

                lb_one_hot = t.zeros(B*K,N).to(logits.device)
                label_exp = label.unsqueeze(1).expand(-1,K).reshape(-1)
                topk_idx = topk_idx.view(-1)
                idx = t.arange(lb_one_hot.shape[0])

                if mixup_lam is None:
                    lb_one_hot[idx, label_exp] = 0.5
                    lb_one_hot[idx, topk_idx] += 0.5
                else:
                    lb_one_hot[idx, topk_idx] += 0.5
                    y_a_exp = y_a.unsqueeze(1).expand(-1,K).reshape(-1)
                    y_b_exp = y_b.unsqueeze(1).expand(-1,K).reshape(-1)
                    lb_one_hot[idx, y_a_exp] += mixup_lam * 0.5
                    lb_one_hot[idx, y_b_exp] += (1.-mixup_lam) * 0.5

                lb_one_hot = lb_one_hot.detach().reshape(B,K,N)
                n_valid = B*K
            
            elif 'word_sim' in self.variant:
                assert self.word_emb_sim is not None
                lb_one_hot = self.word_emb_sim[label]
                ignore = label.eq(self.lb_ignore)
                n_valid = ignore.eq(0).sum()
                idx = t.arange(label.shape[0])
                if self.norm_type == 'l1':
                    lb_one_hot[idx, label] = 0.0
                    lb_one_hot = F.normalize(lb_one_hot, p=1.0, dim=-1)
                elif self.norm_type == 'softmax':
                    lb_one_hot[idx, label] = float('-inf')
                    lb_one_hot /= self.temp
                    lb_one_hot = F.softmax(lb_one_hot, dim=-1)
                lb_one_hot *= self.lb_smooth
                lb_one_hot[idx, label] = 1.0-self.lb_smooth
                lb_one_hot = lb_one_hot.detach()
            
            else:
                raise ValueError
        
        logs = self.log_softmax(logits)
        loss = -t.sum(logs * lb_one_hot, dim=-1)
        
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


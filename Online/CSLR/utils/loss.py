from multiprocessing.sharedctypes import Value
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import math


class LabelSmoothCE(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100, word_emb_tab=None, norm_type='softmax', temp=1.0, 
                variant=None, cls_weight=None, aug_weight=None, bag_size=6, num_instance=4):
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
        self.cls_weight = cls_weight
        self.aug_weight = aug_weight
        self.bag_size = bag_size
        self.num_instance = num_instance


    def forward(self, logits, label, topk_idx=None, mixup_lam=None, y_a=None, y_b=None,
                        fused_mixup_lam=None, fused_y_a=None, fused_y_b=None, fc_weight=None, **kwargs):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        bag_labels = kwargs.pop('bag_labels', None)
        iou_labels = kwargs.pop('iou_labels', None)
        bag_loss = kwargs.pop('bag_loss', False)
        bag_logits = kwargs.pop('bag_logits', None)
        with t.no_grad():
            # print(self.variant)
            if self.variant is None or iou_labels is None:
                num_classes = logits.size(1)
                label = label.clone().detach()
                ignore = label.eq(self.lb_ignore)
                n_valid = ignore.eq(0).sum()
                label[ignore] = 0
                lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / (num_classes-1)
                lb_one_hot = t.empty_like(logits).fill_(
                    lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
            
            elif 'iou_soft' in self.variant and iou_labels is not None:
                assert len(iou_labels) == logits.shape[0]
                B = len(iou_labels)
                x, y, iou = [], [], []
                for i in range(B):
                    num_gls = iou_labels[i][0].shape[0]
                    if 'only5' in self.norm_type and num_gls==1:
                        continue
                    x.append(i*t.ones(num_gls).long())
                    y.append(iou_labels[i][0])
                    iou.append(iou_labels[i][1])
                if len(x)>0:
                    x, y, iou = t.cat(x, dim=0), t.cat(y, dim=0), t.cat(iou, dim=0)
                if 'l1' in self.norm_type:
                    if 'only5' in self.norm_type:
                        num_classes = logits.size(1)
                        label = label.clone().detach()
                        lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / (num_classes-1)
                        lb_one_hot = t.empty_like(logits).fill_(
                            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
                        if type(x) == t.Tensor:
                            lb_one_hot[x] = 0
                    else:
                        lb_one_hot = t.zeros_like(logits)
                    if type(x) == t.Tensor:
                        lb_one_hot[x,y] = iou
                    lb_one_hot = F.normalize(lb_one_hot, p=1.0, dim=-1)
                elif 'softmax' in self.norm_type:
                    if 'only5' in self.norm_type:
                        num_classes = logits.size(1)
                        label = label.clone().detach()
                        lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / (num_classes-1)
                        lb_one_hot = t.empty_like(logits).fill_(
                            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
                        if type(x) == t.Tensor:
                            lb_one_hot[x] = 0
                    else:
                        lb_one_hot = -float('inf')*t.ones_like(logits)
                    if type(x) == t.Tensor:
                        if 'only5' in self.norm_type:
                            lb_one_hot[x,y] = t.exp(iou)/self.temp
                        else:
                            lb_one_hot[x,y] = iou
                    if 'only5' in self.norm_type:
                        lb_one_hot = F.normalize(lb_one_hot, p=1.0, dim=-1)
                    else:
                        lb_one_hot = F.softmax(lb_one_hot/self.temp, dim=-1)
                # print(lb_one_hot)
                n_valid = B

            elif 'dual' in self.variant:
                # assert topk_idx is not None
                B,K,N = logits.shape
                label = label.clone().detach()

                lb_one_hot = t.zeros(B*K,N).to(logits.device)
                label_exp = label.unsqueeze(1).expand(-1,K).reshape(-1)
                topk_idx = topk_idx.view(-1)
                idx = t.arange(lb_one_hot.shape[0])

                if 'word_sim' in self.variant:
                    word_sim = self.word_emb_sim.index_select(0, label_exp)  #[BK,N]
                    lb_one_hot[idx, label_exp] = 1.0
                    lb_one_hot[idx, topk_idx] += word_sim[idx, topk_idx]
                    lb_one_hot = t.clamp(lb_one_hot, max=1.0)
                    lb_one_hot = F.normalize(lb_one_hot, p=1.0, dim=-1)
                elif mixup_lam is None:
                    lb_one_hot[idx, label_exp] = 0.5
                    lb_one_hot[idx, topk_idx] += 0.5
                else:
                    if 'mixup' in self.variant:
                        # mixup_lam comes from vision-language
                        lb_one_hot[idx, label_exp] += mixup_lam.reshape(-1)
                        lb_one_hot[idx, topk_idx] += (1.-mixup_lam).reshape(-1)
                    else:
                        if fused_mixup_lam is not None:
                            lb_one_hot[idx, topk_idx] += 0.5
                            y_a_exp = y_a.unsqueeze(1).expand(-1,K).reshape(-1)
                            y_b_exp = y_b.unsqueeze(1).expand(-1,K).reshape(-1)
                            fused_y_a_exp = fused_y_a.unsqueeze(1).expand(-1,K).reshape(-1)
                            fused_y_b_exp = fused_y_b.unsqueeze(1).expand(-1,K).reshape(-1)
                            lb_one_hot[idx, y_a_exp] += fused_mixup_lam * mixup_lam * 0.5
                            lb_one_hot[idx, y_b_exp] += fused_mixup_lam * (1.-mixup_lam) * 0.5
                            lb_one_hot[idx, fused_y_a_exp] += (1.-fused_mixup_lam) * mixup_lam * 0.5
                            lb_one_hot[idx, fused_y_b_exp] += (1.-fused_mixup_lam) * (1.-mixup_lam) * 0.5
                            # print(fused_mixup_lam)
                        else:
                            # mixup_lam comes from vision-vision
                            lb_one_hot[idx, topk_idx] += 0.5
                            # print(idx.shape, topk_idx.shape)
                            y_a_exp = y_a.unsqueeze(1).expand(-1,K).reshape(-1)
                            y_b_exp = y_b.unsqueeze(1).expand(-1,K).reshape(-1)
                            lb_one_hot[idx, y_a_exp] += mixup_lam * 0.5
                            lb_one_hot[idx, y_b_exp] += (1.-mixup_lam) * 0.5

                lb_one_hot = lb_one_hot.detach().reshape(B,K,N)
                n_valid = B*K
            
            elif 'word_sim' in self.variant:
                head_name = kwargs.pop('head_name', None)
                assert self.word_emb_sim is not None
                lb_one_hot = self.word_emb_sim[label]
                ignore = label.eq(self.lb_ignore)
                n_valid = ignore.eq(0).sum()
                idx = t.arange(label.shape[0])
                if self.norm_type == 'l1':
                    if 'nosmooth' not in self.variant:
                        lb_one_hot[idx, label] = 0.0
                    lb_one_hot = F.normalize(lb_one_hot, p=1.0, dim=-1)
                elif self.norm_type == 'softmax':
                    if 'nosmooth' not in self.variant:
                        lb_one_hot[idx, label] = float('-inf')
                    lb_one_hot /= self.temp
                    lb_one_hot = F.softmax(lb_one_hot, dim=-1)
                if 'xmodal' not in self.variant or head_name not in ['rgb', 'keypoint']:
                    lb_one_hot *= self.lb_smooth
                    lb_one_hot[idx, label] = 1.0-self.lb_smooth
                lb_one_hot = lb_one_hot.detach()

                if 'xmodal' in self.variant and head_name in ['rgb', 'keypoint']:
                    rgb_logits = kwargs.pop('rgb_logits', None)
                    keypoint_logits = kwargs.pop('keypoint_logits', None)
                    fuse_logits = kwargs.pop('fuse_logits', None)
                    if head_name == 'rgb':
                        xmodal_logits = keypoint_logits
                    elif head_name == 'keypoint':
                        xmodal_logits = rgb_logits
                    elif head_name == 'fuse':
                        xmodal_logits = (F.softmax(rgb_logits, dim=-1) + F.softmax(keypoint_logits, dim=-1)).log()
                    
                    temp_extra = self.variant.split('_')[-4]
                    try:
                        temp_extra = float(temp_extra)
                    except:
                        temp_extra = 1.0

                    if 'nonorm' in self.variant:
                        lb_one_hot = self.lb_smooth*lb_one_hot + F.softmax(xmodal_logits, dim=-1)
                        lb_one_hot[idx, label] = 1.0-self.lb_smooth
                    
                    if 'nosmooth' not in self.variant:
                        xmodal_logits[idx, label] = float('-inf')
                    xmodal_prob = F.softmax(xmodal_logits, dim=-1)

                    if 'plus' in self.variant:
                        coef = self.variant.split('_')[4]
                        try:
                            coef = float(coef)
                        except:
                            coef = 1.0
                        lb_one_hot = lb_one_hot + coef * xmodal_prob
                    elif 'mul' in self.variant:
                        lb_one_hot = lb_one_hot * xmodal_prob
                    
                    if 'norm_l1' in self.variant:
                        if 'nosmooth' not in self.variant:
                            lb_one_hot[idx, label] = 0.0
                        lb_one_hot = F.normalize(lb_one_hot, p=1.0, dim=-1)
                    elif 'norm_softmax' in self.variant:
                        if 'nosmooth' not in self.variant:
                            lb_one_hot[idx, label] = float('-inf')
                        lb_one_hot = F.softmax(lb_one_hot/temp_extra, dim=-1)
                    
                    if 'nonorm' not in self.variant and 'nosmooth' not in self.variant:
                        lb_one_hot *= self.lb_smooth
                        lb_one_hot[idx, label] = 1.0-self.lb_smooth

                elif self.variant == 'word_sim_vis':
                    assert fc_weight is not None
                    fc_weight_sim = t.matmul(F.normalize(fc_weight, dim=-1), F.normalize(fc_weight, dim=-1).T)  #[N,N]
                    lb_one_hot_vis = fc_weight_sim[label]
                    lb_one_hot_vis[idx, label] = float('-inf')
                    lb_one_hot_vis /= self.temp
                    lb_one_hot_vis = F.softmax(lb_one_hot_vis, dim=-1)
                    lb_one_hot_vis *= self.lb_smooth
                    lb_one_hot_vis[idx, label] = 1.0-self.lb_smooth
                    lb_one_hot_vis = lb_one_hot_vis.detach()
                    lb_one_hot = 0.5*lb_one_hot + 0.5*lb_one_hot_vis
            
            else:
                raise ValueError
        
        logs = self.log_softmax(logits)
        loss = -t.sum(logs * lb_one_hot, dim=-1)
        # loss[ignore] = 0
        if self.reduction == 'mean':
            # if self.cls_weight is None:
            #     loss = loss.sum() / n_valid
            # else:
            #     self.cls_weight = self.cls_weight.to(loss.device)
            #     w = self.cls_weight[label]
            #     loss = (loss*w).sum() / w.sum()
            # aug = kwargs.pop('aug', None)
            # if self.aug_weight and aug is not None:
            #     w = t.ones(logs.shape[0]).to(logs.device)
            #     w[aug==1] = self.aug_weight
            #     loss = (loss*w).sum() / n_valid
            # else:
            if bag_labels is not None:
                # print(bag_labels, self.num_instance)
                tot_loss = t.zeros(self.num_instance).to(loss.device)
                for i in range(bag_labels.shape[0]//self.bag_size):
                    idx = t.argsort(loss[i*self.bag_size:(i+1)*self.bag_size])
                    tot_loss += loss[i*self.bag_size:(i+1)*self.bag_size][idx[:self.num_instance]]
                loss = tot_loss.sum() / (bag_labels.shape[0]//self.bag_size*self.num_instance)
                # print(loss)
                if bag_logits is not None or bag_loss in ['avg_prob', 'only_bag']:
                    if bag_logits is not None:
                        prob = bag_logits.softmax(dim=-1)
                        # print('bag_logits')
                    else:
                        # print('avg_prob')
                        prob = logits.softmax(dim=-1)
                    n_bag = bag_labels.shape[0]//self.bag_size
                    prob = prob.view(n_bag, self.bag_size, -1)
                    prob = prob.mean(dim=1)
                    cleaned_lb_one_hot = []
                    for i in range(n_bag):
                        cleaned_lb_one_hot.append(lb_one_hot[i*self.bag_size])
                    cleaned_lb_one_hot = t.stack(cleaned_lb_one_hot, dim=0)
                    if bag_loss == 'only_bag':
                        loss = - t.sum(prob.log() * cleaned_lb_one_hot, dim=-1).sum()/n_bag
                    else:
                        loss = loss - t.sum(prob.log() * cleaned_lb_one_hot, dim=-1).sum()/n_bag

            else:
                loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BCEwithWordSim(nn.Module):
    def __init__(self, reduction='mean', word_emb_tab=None):
        super(BCEwithWordSim, self).__init__()
        self.reduction = reduction
        if word_emb_tab is not None:
            self.word_emb_sim = t.matmul(F.normalize(word_emb_tab, dim=-1), F.normalize(word_emb_tab, dim=-1).T)  #[N,N]
        
    def forward(self, logits, label):
        B, N = logits.shape
        word_sim = self.word_emb_sim.index_select(0, label)
        word_sim = word_sim.clone().detach()  #[B,N]

        # https://stackoverflow.com/questions/48951109/keras-custom-binary-cross-entropy-loss-function-get-nan-as-output-for-loss
        loss = t.clamp(logits, min=0) - logits * word_sim + t.log(1 + t.exp(-t.abs(logits)))

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss
    

class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".
        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = t.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: distributions with batch*seq_len x vocab_size
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == targets.shape
            )
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )
        return loss
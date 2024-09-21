# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

from external_metrics import Rouge, sacrebleu
import numpy as np
import pickle

WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4


def chrf(references, hypotheses):
    """
    Character F-score from sacrebleu
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return (
        sacrebleu.corpus_chrf(hypotheses=hypotheses, references=references).score * 100
    )


def bleu(references, hypotheses, level='word'):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    if level=='char':
        #split word
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(r)) for r in hypotheses]
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores


def token_accuracy(references, hypotheses, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    split_char = " " if level in ["word", "bpe"] else ""
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp.split(split_char), ref.split(split_char)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens) * 100 if all_tokens > 0 else 0.0


def sequence_accuracy(references, hypotheses):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum(
        [1 for (hyp, ref) in zip(hypotheses, references) if hyp == ref]
    )
    return (correct_sequences / len(hypotheses)) * 100 if hypotheses else 0.0


def rouge(references, hypotheses, level='word'):
    if level=='char':
        hyp = [list(x) for x in hypotheses]
        ref = [list(x) for x in references]
    else:
        hyp = [x.split() for x in hypotheses]
        ref = [x.split() for x in references]
    a = Rouge.rouge([' '.join(x) for x in hyp], [' '.join(x) for x in ref])
    return a['rouge_l/f_score']*100


def wer_list_per_sen(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    wer = num = 0
    for r, h in zip(references, hypotheses):
        res = wer_single(r=r, h=h)
        total_error += res["num_err"]
        total_del += (res["num_del"] / res["num_ref"])
        total_ins += (res["num_ins"] / res["num_ref"])
        total_sub += (res["num_sub"] / res["num_ref"])
        total_ref_len += res["num_ref"]
        wer += (res["num_err"] / res["num_ref"])
        num += 1
        # print(res["num_ref"])

    wer = (wer / num) * 100
    del_rate = (total_del / num) * 100
    ins_rate = (total_ins / num) * 100
    sub_rate = (total_sub / num) * 100

    return {
        "wer": wer,
        "del_rate": del_rate,
        "ins_rate": ins_rate,
        "sub_rate": sub_rate,
        "del":total_del,
        "ins":total_ins,
        "sub":total_sub,
        "ref_len":total_ref_len,
        "error":total_error,
    }


def wer_single(r, h):
    r = r.strip().split()
    h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)

    num_cor = np.sum([s == "C" for s in alignment['align_lst']])
    num_del = np.sum([s == "D" for s in alignment['align_lst']])
    num_ins = np.sum([s == "I" for s in alignment['align_lst']])
    num_sub = np.sum([s == "S" for s in alignment['align_lst']])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }


def edit_distance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                # d[0][j] = j
                d[0][j] = j * WER_COST_INS
            elif j == 0:
                d[i][0] = i * WER_COST_DEL
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_INS
                delete = d[i - 1][j] + WER_COST_DEL
                d[i][j] = min(substitute, insert, delete)
    return d


def get_alignment(r, h, d):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)

    alignlist = []
    align_ref_lst = []
    align_hyp_lst = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            align_ref_lst.append(r[x - 1])
            align_hyp_lst.append(h[y - 1])
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            align_ref_lst.append(r[x - 1])
            align_hyp_lst.append(h[y - 1])
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("I")
            align_ref_lst.append("*" * len(h[y - 1]))
            align_hyp_lst.append(h[y - 1])
            x = max(x, 0)
            y = max(y - 1, 0)
        else:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("D")
            align_ref_lst.append(r[x - 1])
            align_hyp_lst.append("*" * len(r[x - 1]))
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        {"align_ref_lst": align_ref_lst[::-1], "align_hyp_lst": align_hyp_lst[::-1], "align_lst": alignlist[::-1]},
        {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
    )


def clean_tvb(s):
    # clean unwanted glosses
    op = []
    for t in s.split():
        if '<' in t and '>' in t:
            continue
        op.append(t)
    return ' '.join(op)


if __name__ == '__main__':
    hyp = ['a b c', 'd e f']
    ref = ['a b', 'd d f']
    hyp = [clean_tvb(h) for h in hyp]
    ref = [clean_tvb(r) for r in ref]
    wer = wer_list_per_sen(ref, hyp)
    print(wer['wer'], wer['sub_rate'], wer['ins_rate'], wer['del_rate'])

    with open('./tvb_results.pkl', 'rb') as f:
        res = pickle.load(f)
    ref = [clean_tvb(res[n]['gls_ref']) for n in res]
    hyp = [clean_tvb(res[n]['ensemble_last_gls_hyp']) for n in res]
    wer = wer_list_per_sen(ref, hyp)
    print(wer['wer'], wer['sub_rate'], wer['ins_rate'], wer['del_rate'])

    # hyp = ['abcdefg', 'hijklmn']
    # ref = ['abcd', 'hijk']
    # bleu_dict = bleu(ref, hyp, level='char')
    # rouge_score = rouge(ref, hyp, level='char')
    # for k,v in bleu_dict.items():
    #     print(k, v)
    # print('ROUGE: {:.2f}'.format(rouge_score))
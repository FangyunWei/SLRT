from itertools import groupby
import re


def clean_phoenix_2014(prediction):
    # TODO (Cihan): Python version of the evaluation script provided
    #  by the phoenix2014 dataset (not phoenix2014t). This should work
    #  as intended but further tests are required to make sure it is
    #  consistent with the bash/sed based clean up script.

    prediction = prediction.strip()
    prediction = re.sub(r"loc-", "", prediction)
    prediction = re.sub(r"cl-", "", prediction)
    prediction = re.sub(r"qu-", "", prediction)
    prediction = re.sub(r"poss-", "", prediction)
    prediction = re.sub(r"lh-", "", prediction)
    prediction = re.sub(r"S0NNE", "SONNE", prediction)
    prediction = re.sub(r"HABEN2", "HABEN", prediction)
    prediction = re.sub(r"__EMOTION__", "", prediction)
    prediction = re.sub(r"__PU__", "", prediction)
    prediction = re.sub(r"__LEFTHAND__", "", prediction)
    prediction = re.sub(r"WIE AUSSEHEN", "WIE-AUSSEHEN", prediction)
    prediction = re.sub(r"ZEIGEN ", "ZEIGEN-BILDSCHIRM ", prediction)
    prediction = re.sub(r"ZEIGEN$", "ZEIGEN-BILDSCHIRM", prediction)
    prediction = re.sub(r"^([A-Z]) ([A-Z][+ ])", r"\1+\2", prediction)
    prediction = re.sub(r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 ", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]SCH) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]NN) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) (NN[ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z])$", r"\1+\2", prediction)
    prediction = re.sub(r"([A-Z][A-Z])RAUM", r"\1", prediction)
    prediction = re.sub(r"-PLUSPLUS", "", prediction)
    prediction = re.sub(r" +", " ", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r" +", " ", prediction)

    assert not re.search("__LEFTHAND__", prediction)
    assert not re.search("__EPENTHESIS__", prediction)
    assert not re.search("__EMOTION__", prediction)

    # Remove white spaces and repetitions
    prediction = " ".join(
        " ".join(i[0] for i in groupby(prediction.split(" "))).split()
    )
    prediction = prediction.strip()

    return prediction


def clean_phoenix_2014_trans(prediction):
    """
    I think the format of ground truth annotation is already well processed.
    Some parts of predicted gloss sequence might be processed using this func. e.g. combine single char with +
    """
    prediction = prediction.strip()
    prediction = re.sub(r"__LEFTHAND__", "", prediction) #not in gls.vocab
    prediction = re.sub(r"__EPENTHESIS__", "", prediction) # not in gls.vocab
    prediction = re.sub(r"__EMOTION__", "", prediction) #not in gls.vocab
    prediction = re.sub(r"\b__[^_ ]*__\b", "", prediction) # not in gls.vocab
    prediction = re.sub(r"\bloc-([^ ]*)\b", r"\1", prediction) # not in gls.vocab (remove loc-)
    prediction = re.sub(r"\bcl-([^ ]*)\b", r"\1", prediction) # not in gls.vocab (remove cl-)
    prediction = re.sub(r"\b([^ ]*)-PLUSPLUS\b", r"\1", prediction) # not in gls.vocab (remove -plusplus)
    prediction = re.sub(r"\b([A-Z][A-Z]*)RAUM\b", r"\1", prediction) #only \bRAUM\b exists in gls annotations (remove RAUM)
    prediction = re.sub(r"WIE AUSSEHEN", "WIE-AUSSEHEN", prediction) # not in gls.vocab
    #combine single character with '+'
    prediction = re.sub(r"^([A-Z]) ([A-Z][+ ])", r"\1+\2", 
            prediction) #"A B+","A B "  -> "A+B+" "A+B" #must be at the start of the string
    prediction = re.sub(r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 ", prediction) # "+A B " -> "A+B"
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction) # " A B "->" A+B "
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction) # " 
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction) # "REPEAT?" " A B C "  --> " A+B+C "
    prediction = re.sub(r"([ +]SCH) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]NN) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) (NN[ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z])$", r"\1+\2", prediction)
    prediction = re.sub(r" +", " ", prediction) #remove single +
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)## 把重复出现的 连续大写字母 合为一个 e.g. ‘BBC BBC BBC ’ -> 'BBC
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction) ##???
    prediction = re.sub(r" +", " ", prediction) #remove lonely +  

    # Remove white spaces and repetitions
    # remove white spaces? split()!=split(" ")
    prediction = " ".join(
        " ".join(i[0] for i in groupby(prediction.split(" "))).split()
    )
    prediction = prediction.strip()

    return prediction

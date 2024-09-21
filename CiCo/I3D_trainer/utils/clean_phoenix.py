# From Necati Cihan Camgoz (https://github.com/neccam/slt/blob/master/signjoey/phoenix_utils/phoenix_cleanup.py)

from itertools import groupby
import re


def clean_phoenix_2014(prediction):

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

    prediction = prediction.strip()
    prediction = re.sub(r"__LEFTHAND__", "", prediction)
    prediction = re.sub(r"__EPENTHESIS__", "", prediction)
    prediction = re.sub(r"__EMOTION__", "", prediction)
    prediction = re.sub(r"\b__[^_ ]*__\b", "", prediction)
    prediction = re.sub(r"\bloc-([^ ]*)\b", r"\1", prediction)
    prediction = re.sub(r"\bcl-([^ ]*)\b", r"\1", prediction)
    prediction = re.sub(r"\b([^ ]*)-PLUSPLUS\b", r"\1", prediction)
    prediction = re.sub(r"\b([A-Z][A-Z]*)RAUM\b", r"\1", prediction)
    prediction = re.sub(r"WIE AUSSEHEN", "WIE-AUSSEHEN", prediction)
    prediction = re.sub(r"^([A-Z]) ([A-Z][+ ])", r"\1+\2", prediction)
    prediction = re.sub(r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 ", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]SCH) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]NN) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) (NN[ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z])$", r"\1+\2", prediction)
    prediction = re.sub(r" +", " ", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r" +", " ", prediction)

    # Remove white spaces and repetitions
    prediction = " ".join(
        " ".join(i[0] for i in groupby(prediction.split(" "))).split()
    )
    prediction = prediction.strip()

    return prediction

import os
import pickle as pkl


def get_classification_params(pretrained_dict, arch="InceptionI3d"):
    if arch == "InceptionI3d":
        pretrained_logits_weight = pretrained_dict["module.logits.conv3d.weight"]
        pretrained_logits_bias = pretrained_dict["module.logits.conv3d.bias"]
    else:
        print(f"Did not specify classification layer for {arch}")
        exit()
    return pretrained_logits_weight, pretrained_logits_bias


def find_correspondence_asl_bsl(asl_dataset="wlasl", bsl_pkl="misc/bsl1k/bsl1k_vocab.pkl"):
    print(
        f"Finding correspondence between {asl_dataset} ASL and {bsl_pkl} BSL"
    )
    words_asl = pkl.load(
        open(os.path.join("data", asl_dataset, "info/info.pkl"), "rb")
    )["words"]
    words_bsl = pkl.load(open(bsl_pkl, "rb"))["words"]
    # Find the indices of the corresponding bsl->asl words
    bsl_to_asl = {}
    for bi, w in enumerate(words_bsl):
        if w in words_asl:
            ai = words_asl.index(w)
            bsl_to_asl[bi] = ai
    return bsl_to_asl


def init_bsl_with_asl(model, asl_logits_weight, asl_logits_bias, bsl_to_asl):
    print("Initializing BSL weights with ASL")
    # Initialize the weights with the std of the pretrained network
    # stdv = 1. / math.sqrt(self.weight.size(1))  # normally this was used
    stdv = asl_logits_weight.std()
    model.module.logits.conv3d.weight.data.uniform_(-stdv, stdv)
    model.module.logits.conv3d.bias.data.uniform_(-stdv, stdv)
    # If the BSL layer replaces the ASL layer
    for c_bsl, c_asl in bsl_to_asl.items():
        # Replace the weight and bias from the corresponding word
        # model.module.logits.conv3d.weight is [1350, 1024, 1, 1, 1]
        model.module.logits.conv3d.weight[c_bsl].data.copy_(asl_logits_weight[c_asl])
        model.module.logits.conv3d.bias[c_bsl].data.copy_(asl_logits_bias[c_asl])
    # Should we freeze the corresponding vectors' gradients?
    return model


def init_asl_with_bsl(model, bsl_logits_weight, bsl_logits_bias, bsl_to_asl):
    print("Initializing ASL weights with BSL")
    # Initialize the weights with the std of the pretrained network
    # stdv = 1. / math.sqrt(self.weight.size(1))  # normally this was used
    stdv = bsl_logits_weight.std()
    model.module.logits.conv3d.weight.data.uniform_(-stdv, stdv)
    model.module.logits.conv3d.bias.data.uniform_(-stdv, stdv)
    # If the BSL layer replaces the ASL layer
    for c_bsl, c_asl in bsl_to_asl.items():
        # Replace the weight and bias from the corresponding word
        # model.module.logits.conv3d.weight is [1350, 1024, 1, 1, 1]
        model.module.logits.conv3d.weight[c_asl].data.copy_(bsl_logits_weight[c_bsl])
        model.module.logits.conv3d.bias[c_asl].data.copy_(bsl_logits_bias[c_bsl])
    return model


def init_cross_language(
    init_str, model, pretrained_w, pretrained_b, asl_dataset, bsl_pkl
):
    bsl_to_asl = find_correspondence_asl_bsl(asl_dataset=asl_dataset, bsl_pkl=bsl_pkl)
    if init_str == "bsl_with_asl":
        return init_bsl_with_asl(model, pretrained_w, pretrained_b, bsl_to_asl)
    elif init_str == "asl_with_bsl":
        return init_asl_with_bsl(model, pretrained_w, pretrained_b, bsl_to_asl)
    else:
        print(f"Not recognized cross language argument {init_str}")
        exit()

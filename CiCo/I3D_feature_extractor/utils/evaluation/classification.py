def performance(output, target, topk=[1, 5]):
    """
    Returns the accuracy at top-k over a batch
        output: [batchsize x num_classes] torch matrix, output of the model
        target: [batchsize] torch vector, indices of the GT classes
        topk: list of k values
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    acc = []
    for k in topk:
        pos = 0.0
        for b in range(batch_size):
            if target[b] in pred[b, :k]:
                pos += 1
        acc.append(pos / batch_size)
    return acc

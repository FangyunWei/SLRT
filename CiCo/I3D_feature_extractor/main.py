"""Main code driver
"""

import logging
import os
import sys
import time
from pathlib import Path

import humanize
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.optim

import evaluate
import models
import opts
from datasets.multidataloader import MultiDataLoader
from epoch import do_epoch
from utils.logger import Logger, savefig, setup_verbose_logging
from utils.misc import (adjust_learning_rate, load_checkpoint,
                        load_checkpoint_flexible, mkdir_p, save_checkpoint)

def main(args):
    # Seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)

    if args.featurize_mode:
        msg = "To perform featurization, use evaluation mode"
        assert args.evaluate and args.evaluate_video, msg
        msg = (
            f"Until we fully understand the implications of multi-worker caching, we "
            f"should avoid using multiple workers (requested {args.workers})"
        )
        assert args.workers <= 1, msg

    # create checkpoint dir
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    # Overload print statement to log to file
    setup_verbose_logging(Path(args.checkpoint))
    logger_name = "train" if not args.evaluate else "eval"
    plog = logging.getLogger(logger_name)

    opts.print_args(args)
    opts.save_args(args, save_folder=args.checkpoint)

    if not args.debug:
        plt.switch_backend("agg")

    # create model
    plog.info(f"==> creating model '{args.arch}', out_dim={args.num_classes}")
    if args.arch == "InceptionI3d":
        model = models.__dict__[args.arch](
            num_classes=args.num_classes,
            spatiotemporal_squeeze=True,
            final_endpoint="Logits",
            name="inception_i3d",
            in_channels=3,
            dropout_keep_prob=0.5,
            num_in_frames=args.num_in_frames,
            include_embds=args.include_embds,
        )
        if args.save_features:
            msg = "Set --include_embds 1 to save_features"
            assert args.include_embds, msg
    elif args.arch == "Pose2Sign":
        model = models.Pose2Sign(num_classes=args.num_classes,)
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes,)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # adjust for opts for multi-gpu training. Note that we also apply warmup to the
    # learning rate. Can technically remove this if-statement, but leaving for now
    # to make the change explicit.
    if args.num_gpus > 1:
        num_gpus = torch.cuda.device_count()
        msg = f"Requested {args.num_gpus}, but {num_gpus} were visible"
        assert num_gpus == args.num_gpus, msg
        args.train_batch = args.train_batch * args.num_gpus
        args.test_batch = args.test_batch * args.num_gpus
        device_ids = list(range(args.num_gpus))
        args.lr = args.lr * args.num_gpus
    else:
        device_ids = [0]

    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    tic = time.time()
    title = f"{args.datasetname} - {args.arch}"
    if args.resume:
        if os.path.isfile(args.resume):
            plog.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = load_checkpoint(args.resume)
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"]
            plog.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
            logger = Logger(
                os.path.join(args.checkpoint, "log.txt"), title=title, resume=True
            )
            del checkpoint
        else:
            plog.info(f"=> no checkpoint found at '{args.resume}'")
            raise ValueError(f"Checkpoint not found at {args.resume}!")
    else:
        logger = Logger(os.path.join(args.checkpoint, "log.txt"), title=title)
        logger_names = ["Epoch", "LR", "train_loss", "val_loss"]
        for p in range(0, args.nloss - 1):
            logger_names.append("train_loss%d" % p)
            logger_names.append("val_loss%d" % p)
        for p in range(args.nperf):
            logger_names.append("train_perf%d" % p)
            logger_names.append("val_perf%d" % p)

        logger.set_names(logger_names)

    if args.pretrained:
        load_checkpoint_flexible(model, optimizer, args, plog)

    param_count = humanize.intword(sum(p.numel() for p in model.parameters()))
    plog.info(f"    Total params: {param_count}")
    duration = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - tic))
    plog.info(f"Loaded parameters for model in {duration}")

    mdl = MultiDataLoader(
        train_datasets=args.datasetname, val_datasets=args.datasetname,
    )

    save_dir=args.save_dir


    train_loader, val_loader, meanstd = mdl._get_loaders(args)

    train_mean = meanstd[0]
    train_std = meanstd[1]
    val_mean = meanstd[2]
    val_std = meanstd[3]


    # Define criterion
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    criterion = criterion.to(device)

    if args.evaluate or args.evaluate_video:
        plog.info("\nEvaluation only")
        loss, acc = do_epoch(
            "val",
            val_loader,
            model,
            criterion,
            num_classes=args.num_classes,
            debug=args.debug,
            checkpoint=args.checkpoint,
            mean=val_mean,
            std=val_std,
            feature_dim=args.feature_dim,
            save_logits=True,
            save_features=args.save_features,
            num_figs=args.num_figs,
            topk=args.topk,
            save_dir=os.path.join(save_dir,args.split),
            epochno=args.rank,
        )
        if args.featurize_mode:
            plog.info(f"Featurizing without metric evaluation")
            return

        # Summarize/save results
        evaluate.evaluate(args, val_loader.dataset, plog)

        logger_epoch = [0, 0]
        for p in range(len(loss)):
            logger_epoch.append(float(loss[p].avg))
            logger_epoch.append(float(loss[p].avg))
        for p in range(len(acc)):
            logger_epoch.append(float(acc[p].avg))
            logger_epoch.append(float(acc[p].avg))
        # append logger file
        logger.append(logger_epoch)

        return



if __name__ == "__main__":
    args = opts.parse_opts(argv=sys.argv[1:])
    main(args)

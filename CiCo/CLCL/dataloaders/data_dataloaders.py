import torch
from torch.utils.data import DataLoader


from dataloaders.dataloader_H2_retrieval import H2_DataLoader
from dataloaders.dataloader_ph_retrieval import ph_DataLoader
from dataloaders.dataloader_ph_retrieval_train import ph_DataLoader_train
from dataloaders.dataloader_H2_retrieval_train import H2_DataLoader_train
from dataloaders.dataloader_csl_retrieval_train import csl_DataLoader_train
from dataloaders.dataloader_csl_retrieval import csl_DataLoader





def dataloader_h2s_train(args, tokenizer):
    h2s_dataset = H2_DataLoader_train(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        feature_len=args.feature_len,
        args=args
    )

    if args.distributed==True:
        sampler = torch.utils.data.distributed.DistributedSampler(h2s_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(h2s_dataset)
    dataloader = DataLoader(
        h2s_dataset,
        batch_size=args.batch_size// args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    return dataloader, len(h2s_dataset), sampler


def dataloader_h2s_test(args, tokenizer, subset="test"):
    h2s_testset = H2_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        feature_len=args.feature_len,
        args=args
    )
    dataloader_msrvtt = DataLoader(
        h2s_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(h2s_testset)


def dataloader_ph_train(args, tokenizer):
    ph_train_dataset = ph_DataLoader_train(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        feature_len=args.feature_len,
        args=args
    )

    if args.distributed==True:
        sampler = torch.utils.data.distributed.DistributedSampler(ph_train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(ph_train_dataset)
    dataloader = DataLoader(
        ph_train_dataset,
        batch_size=args.batch_size// args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    return dataloader, len(ph_train_dataset), sampler


def dataloader_ph_test(args, tokenizer, subset="test"):
    hp_testset = ph_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        feature_len=args.feature_len,
        args=args
    )
    dataloader_ph= DataLoader(
        hp_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_ph, len(hp_testset)



def dataloader_csl_train(args, tokenizer):
    train_dataset = csl_DataLoader_train(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        feature_len=args.feature_len,
        args=args
    )

    if args.distributed==True:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size// args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    return dataloader, len(train_dataset), sampler


def dataloader_csl_test(args, tokenizer, subset="test"):
    testset = csl_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        feature_len=args.feature_len,
        args=args
    )
    dataloader= DataLoader(
        testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(testset)


DATALOADER_DICT = {}

DATALOADER_DICT["h2s"] = {"train":dataloader_h2s_train, "dev":dataloader_h2s_test, "test":dataloader_h2s_test}
DATALOADER_DICT["ph"] = {"train":dataloader_ph_train, "dev":dataloader_h2s_test, "test":dataloader_ph_test}
DATALOADER_DICT["csl"] = {"train":dataloader_csl_train, "dev":dataloader_csl_test, "test":dataloader_csl_test}



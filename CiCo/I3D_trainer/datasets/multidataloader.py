import copy

import mock
import torch

import datasets
from beartype import beartype
from datasets.videodataset import VideoDataset


class MultiDataLoader:
    def __init__(self, train_datasets, val_datasets):
        self.datasets = {"train": train_datasets, "val": val_datasets}
        print("Data loading info:")
        print(self.datasets)

    def _get_loaders(self, args):
        common_kwargs = {
            "stride": args.stride,
            "inp_res": args.inp_res,
            "resize_res": args.resize_res,
            "num_in_frames": args.num_in_frames,
            "gpu_collation": args.gpu_collation,
        }
        loaders = {}
        for split, dataset_name in self.datasets.items():
            if not dataset_name:
                dataset = []
                break


            kwargs = copy.deepcopy(common_kwargs)
            if split == "val":
                kwargs.update(
                    {
                        "setname": args.test_set,
                        "evaluate_video": args.evaluate_video,
                    }
                )

            elif dataset_name=="H2S":
                dataset=datasets.H2S(**kwargs)

            elif dataset_name == "phoenix2014":
                dataset = datasets.PHOENIX2014(**kwargs)

            elif dataset_name == "csl":
                dataset = datasets.CSL(**kwargs)
            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name}")
            loaders[split] = dataset

        # TODO: dataloader_train = mock.Mock()
        dataloader_train = loaders["train"]
        dataloader_val = loaders["val"]
        sampler_train = None
        sampler_val = None
        data_shuffle = True

        # Data loading code - set shared kwargs
        loader_kwargs = {"pin_memory": True, "num_workers": args.workers}

        if not args.evaluate_video:
            # Note: to avoid excessive monkey patching, we share a common collation
            # function across all VideoDataset instances. It is therefore important to
            # ensure that collation has no dependency on class attributes
            if isinstance(dataloader_train, torch.utils.data.ConcatDataset):
                train_collate = dataloader_train.datasets[0].collate_fn
            else:
                train_collate = dataloader_train.collate_fn

            train_loader = torch.utils.data.DataLoader(
                dataloader_train,
                batch_size=args.train_batch,
                collate_fn=train_collate,
                sampler=sampler_train,
                shuffle=data_shuffle,
                **loader_kwargs,
            )
        else:
            dataloader_train.mean = None
            dataloader_train.std = None
            train_loader = mock.Mock()

        val_loader = torch.utils.data.DataLoader(
            dataloader_val,
            batch_size=args.test_batch,
            shuffle=False,
            sampler=sampler_val,
            collate_fn=dataloader_val.collate_fn,
            **loader_kwargs,
        )
        meanstd = [
            dataloader_train.mean,
            dataloader_train.std,
            dataloader_val.mean,
            dataloader_val.std,
        ]
        return train_loader, val_loader, meanstd

# A simple tensorboard logger

import getpass
import json
import logging.config
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.misc import Timer

__all__ = ["Logger", "savefig"]


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)


def setup_verbose_logging(save_dir, log_config="utils/logger_config.json",
                          default_level=logging.INFO):
    """Setup logging configuration."""
    print(os.getcwd())
    log_config = Path(log_config)
    print(f"log config: {log_config} exists: {log_config.exists()}")
    if log_config.is_file():
        with open(log_config, "r") as f:
            config = json.load(f)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])
        logging.config.dictConfig(config)
    else:
        print(f"Warning: logging configuration file is not found in {log_config}.")
        logging.basicConfig(level=default_level)
    return config["handlers"]["info_file_handler"]["filename"]


class Logger(object):
    """Save training process to log file with simple plot function."""

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = "" if title is None else title
        self.json_path = os.path.splitext(fpath)[0] + ".json"

        if fpath is not None:
            if resume and Path(fpath).exists() and Path(self.json_path).exists():
                self.file = open(fpath, "r")
                name = self.file.readline()
                self.names = name.rstrip().split("\t")
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split("\t")
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, "a")

                json_data = open(self.json_path).read()
                self.figures = json.loads(json_data)
                # self.json_file.close()

            else:
                self.file = open(fpath, "w")
                self.figures = {}

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write("\t")
            self.numbers[name] = []
            if name not in self.figures.keys():
                if len(name.split("_")) > 1:
                    fig_id = name.split("_")[1]  # take 'loss' if it is 'val_loss'
                else:
                    fig_id = name
                self.figures[fig_id] = {}
                self.figures[fig_id]["data"] = []
                self.figures[fig_id]["layout"] = {"title": fig_id}
        self.file.write("\n")
        self.file.flush()

    def append(self, numbers):
        if not hasattr(self, "names") and getpass.getuser() == "albanie":
            print(
                f"{getpass.getuser()} applying wearily woeful and gloomily glum hack :("
            )
            names = ["Epoch", "LR", "train_loss", "val_loss"]
            guessed_nperf = (len(numbers) - len(names)) // 2
            for ii in range(guessed_nperf):
                names.append("train_perf%d" % ii)
                names.append("val_perf%d" % ii)
            self.set_names(names)

        assert len(self.names) == len(numbers), "Numbers do not match names"
        for index, num in enumerate(numbers):
            self.file.write("{0:.3f}".format(num))
            self.file.write("\t")
            self.numbers[self.names[index]].append(num)
        self.file.write("\n")
        self.file.flush()

        for index, num in enumerate(numbers):
            if len(self.names[index].split("_")) > 1:
                plot_id = self.names[index].split("_")[
                    0
                ]  # take 'val' if it is 'val_loss'
                fig_id = self.names[index].split("_")[
                    1
                ]  # take 'loss' if it is 'val_loss'
            else:
                plot_id = self.names[index]
                fig_id = self.names[index]
            fig_data = self.figures[fig_id]["data"]
            plot = None
            for k, v in enumerate(fig_data):
                if v["name"] == plot_id:
                    plot = v

            if plot is None:
                plot = {"name": plot_id, "x": [], "y": []}
                fig_data.append(plot)

            # Epoch
            plot["x"].append(numbers[0])
            # Value
            plot["y"].append(num)

            self.json_file = open(self.json_path, "w")
            self.json_file.write(json.dumps(self.figures))
            self.json_file.close()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + "(" + name + ")" for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class TensorboardWriter:
    """A second class for interfacing with tensorboard. Derived from the wrapper
    provided with Pytorch-Template by Victor Huang.
    (https://github.com/victoresque/pytorch-template)
    """

    def __init__(self, log_dir):
        self.writer = None
        self.selected_module = ""
        self.writer = SummaryWriter(str(log_dir))
        self.step = 0
        self.mode = ""
        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = Timer()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer.reset()
        else:
            duration = self.timer.check()
            self.add_scalar("steps_per_sec", 1 / duration)

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information
            (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = f"{tag}/{self.mode}"
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step()
            # for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                msg = "type object '{}' has no attribute '{}'"
                raise AttributeError(msg.format(self.selected_module, name))
            return attr

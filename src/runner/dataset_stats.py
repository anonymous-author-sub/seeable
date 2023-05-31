import json
import os
import random
from itertools import product
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from src.utils import Logger
from src.utils.dataframe import describe, describe_col, get_col_unique


class DatasetStats:
    def __init__(self, cfg, root=None, columns=["label", "fold", "split"]):
        self.cfg = cfg
        self.columns = columns

    def debug_dataset(self, dataset):
        """
            label            fold  split  count  percent
        0       0         youtube   test    140     2.33
        1       0         youtube  train    720    12.00
        2       0         youtube    val    140     2.33
        3       1       Deepfakes   test    140     2.33
        4       1       Deepfakes  train    720    12.00
        5       1       Deepfakes    val    140     2.33
        6       1       Face2Face   test    140     2.33
        7       1       Face2Face  train    720    12.00
        8       1       Face2Face    val    140     2.33
        9       1     FaceShifter   test    140     2.33
        10      1     FaceShifter  train    720    12.00
        11      1     FaceShifter    val    140     2.33
        12      1        FaceSwap   test    140     2.33
        13      1        FaceSwap  train    720    12.00
        14      1        FaceSwap    val    140     2.33
        15      1  NeuralTextures   test    140     2.33
        16      1  NeuralTextures  train    720    12.00
        17      1  NeuralTextures    val    140     2.33
        """
        print("Debuging dataset ...")
        # -- per col stats --
        out = describe(dataset.df, columns=self.columns)
        # print("DEBUG: out =", out)

        # -- get a sample of the dataset --
        sample: dict = random.choice(dataset)
        print(sample, "\n")

        # per item stats : cartesian product {label x fold x split}
        DF = dataset.df  # full dataframe
        cols_name: List[str] = self.columns

        # get all possible combination of cols
        # e.g. [(0, 'youtube', 'test'), ..., (1, 'NeuralTextures', 'val')]
        cols_samples = list(product(*[get_col_unique(DF, col) for col in cols_name]))  # type: ignore

        df_header = self.columns + ["count", "percent"]
        df_data = []

        for col_samples in cols_samples:
            # array of bool
            I = np.logical_and.reduce(
                [DF[cn] == cv for (cn, cv) in zip(cols_name, col_samples)]
            )
            sample_size: int = np.count_nonzero(I)
            if sample_size > 0:
                sample_size_p = round(100 * sample_size / len(DF), 2)

                df_row = (*col_samples, sample_size, sample_size_p)
                # print(df_row)
                df_data.append(df_row)

        df = pd.DataFrame(df_data, columns=df_header)
        print(df.to_string(index=False))
        print("*" * 80)

        # save it to file csv
        filename_out = "dataset_stats.csv"
        df.to_csv(filename_out, index=False)

        filename_md = "dataset_stats.md"
        with open(filename_md, "w") as f:
            f.write(df.to_markdown(index=False))

        # get absolute pathlib
        path = Path(filename_out).absolute()
        logger.info(f"Saved to {path.parent}")

        # create a mardown file with the stats
        filename_md_global = "dataset_stats_global.md"
        with open(filename_md_global, "w") as f:
            print(f"total: {len(dataset)} videos", file=f)
            for split in out:
                print(f"{split}", file=f)
                for split_value in out[split]:
                    print("-", f"**{split_value[0]}**", *split_value[1:], file=f)
                print(file=f)

    def run(self):
        logger.info("Running SELECTED...")
        dataset = instantiate(self.cfg.dataset)

        # create a directory with the dataset name
        dir = Path(self.cfg.dataset.name)
        dir.mkdir(parents=True, exist_ok=True)
        # move to the directory
        os.chdir(str(dir))

        self.debug_dataset(dataset)

        if False:
            print("=" * 30)
            logger.info("Running GOBAL ...")
            dataset.update_selector({})
            self.debug_dataset(dataset)

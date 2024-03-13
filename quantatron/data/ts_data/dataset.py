# --*-- conding:utf-8 --*--
# @Time : 2024/1/17 下午1:18
# @Author : Shi Lei
# @Email : 2055833480@qq.com
# @File : dataset.py
# @Software : PyCharm

from quantatron.data.datasets import NormalDataset
import pandas as pd
from datetime import datetime

import tushare as ts

import os

import torch
import numpy as np
from ..build import DATASET_REGISTRY
from quantatron.config import configurable

# pro = ts.pro_api("6f247516ae8556bd39833c8e6c257c3bfe0e7f33ce48db183cbbdf34")
pro = ts.pro_api()


@DATASET_REGISTRY.register()
class TSDailyDataset(NormalDataset):
    @configurable
    def __init__(self, start_date: str, end_date: str, mode, dir_path, stock_codes=[], length: int=10):
        super(TSDailyDataset, self).__init__(start_date, end_date, mode, dir_path, stock_codes, length)

    @classmethod
    def from_config(cls, cfg):
        return {
            "start_date": cfg.START_DATE,
            "end_date": cfg.END_DATE,
            "mode": cfg.MODE,
            "dir_path": cfg.DIR,
            "stock_codes": list(cfg.STOCK_CODES),
            "length": cfg.LENGTH,
        }

    def download(self, start_date, end_date, code):
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%Y%m%d")
        df = pro.query('daily', ts_code=code, start_date=start_date, end_date=end_date)
        df['date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d').dt.strftime(
            '%Y-%m-%d')
        del df["trade_date"]

        return df

    def get_file_paths(self):
        dir_paths = [os.path.join(self.dir_path, code.replace('.', '_')) for code in self.stock_codes]
        for dir in dir_paths:
            if not os.path.exists(dir):
                os.mkdir(dir)
        self.file_paths = [os.path.join(self.dir_path, code.replace('.', '_') + "/daily.csv") for code in
                           self.stock_codes]

    def __len__(self):
        length = 0
        for code in self.stock_codes:
            df = self.data_dict[code]
            # -self.length 表示单次数据量为10天， -1 表示需要一天数据作为标签
            length += (len(df) - self.length - 1)
        return length

    def __getitem__(self, item):
        length = 0
        code = None

        for code in self.stock_codes:
            df = self.data_dict[code]
            if item < length + (len(df) - self.length - 1):
                break
            length += (len(df) - self.length - 1)
        item = item - length

        data = (self.data_dict[code]).iloc[item:item + self.length]
        del data["date"]
        del data["ts_code"]
        data = torch.tensor(data.to_numpy(), dtype=torch.float32)
        y = self.data_dict[code].iloc[item + self.length][["open", "high", "low", "close"]]
        y = y.astype(float)
        y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        return {"daily": data, "gt": y}


class TSIncomeDataset(NormalDataset):
    def download(self, start_date, end_date, code):
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%Y%m%d")

        df = pro.income(ts_code=code, start_date=start_date, end_date=end_date,
                        fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,diluted_eps')

        df['date'] = pd.to_datetime(df['ann_date'].astype(str), format='%Y%m%d').dt.strftime(
            '%Y-%m-%d')
        del df["ann_date"]

        return df

    def get_file_paths(self):
        dir_paths = [os.path.join(self.dir_path, code.replace('.', '_')) for code in self.stock_codes]
        for dir in dir_paths:
            if not os.path.exists(dir):
                os.mkdir(dir)
        self.file_paths = [os.path.join(self.dir_path, code.replace('.', '_') + "/income.csv") for code in
                           self.stock_codes]

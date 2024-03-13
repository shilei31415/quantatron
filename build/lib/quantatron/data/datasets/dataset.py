# --*-- conding:utf-8 --*--
# @Time : 2024/1/17 上午11:58
# @Author : Shi Lei
# @Email : 2055833480@qq.com
# @File : dataset.py
# @Software : PyCharm

import torch
from torch.utils.data import Dataset

import time
import os
import pandas as pd

from quantatron.utils.logger import setup_logger

from datetime import datetime

logger = setup_logger(name="quantatron.data")

# 年-月-日 月和日保留两位
class NormalDataset(Dataset):
    """
    基础数据集类，具有下载数据，记载数据的基本功能
    其他数据集，如日线数据集等，均继承本类
    """

    def __init__(self, start_date: str, end_date: str, mode, dir_path, stock_codes=[], length=0):
        assert mode in ["offline", "online"]

        self.mode = mode
        self.start_date = start_date
        self.end_date = end_date
        self.dir_path = dir_path
        self.stock_codes = stock_codes

        self.get_file_paths()
        self.data_dict = {}

        self.length = length

        logger.info("loading....")
        start_time = time.time()

        # load data
        self.load()
        for code in self.stock_codes:
            download_request = self.check_local(code)
            if download_request != None and mode == "online":
                start_dt, end_dt = download_request
                tmp_data = self.download(start_dt, end_dt, code)
                self.save(tmp_data, code)

        end_time = time.time()
        logger.info(f"Done({(end_time - start_time):.2f})")

        # remove useless data
        # start_date = datetime.strptime(self.start_date, "%Y%m%d").strftime("%Y-%m-%d")
        # end_date = datetime.strptime(self.end_date, "%Y%m%d").strftime("%Y-%m-%d")
        for code in self.stock_codes:
            df = self.data_dict[code]
            self.data_dict[code] = df[(df["date"] <= end_date)
                                      & (df["date"] >= start_date)]
            self.data_dict[code] = (self.data_dict[code]).sort_values(by="date")

    def get_file_paths(self):
        self.file_paths = [os.path.join(self.dir_path, code.replace('.', '_') + ".csv") for code in self.stock_codes]

    def save(self, tmp_data, code=None):
        # merage
        if code in self.data_dict:
            combined_df = pd.concat([self.data_dict[code], tmp_data])
        else:
            combined_df = tmp_data
        combined_df = combined_df.drop_duplicates(subset=["date"])
        combined_df.sort_values("date", ascending=False, inplace=True)
        self.data_dict[code] = combined_df
        # save
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        if code is None:
            for code, path in zip(self.stock_codes, self.file_paths):
                self.data_dict[code].to_csv(path, index=False)
        else:
            idx = self.stock_codes.index(code)
            path = self.file_paths[idx]
            self.data_dict[code].to_csv(path, index=False)

    def download(self, start_date, end_date, stock_codes):
        pass

    def load(self):
        for code, path in zip(self.stock_codes, self.file_paths):
            try:
                self.data_dict[code] = pd.read_csv(path)
            except Exception as e:
                logger.info("Data loading filed")
                logger.info(e)

    def check_local(self, code):
        if code not in self.data_dict:
            return (self.start_date, self.end_date)
        else:
            dates = self.data_dict[code]["date"]
            # dates = pd.to_datetime(dates.astype(str), format='%Y-%m-%d').dt.strftime('%Y%m%d')
            dates = [str(date) for date in dates]

        if len(dates) < 3:
            return (self.start_date, self.end_date)

        if dates[0] >= self.end_date and dates[-1] <= self.start_date:
            return None
        if dates[0] < self.end_date and dates[-1] > self.start_date:
            return (self.start_date, self.end_date)
        if dates[0] < self.end_date and dates[-1] <= self.start_date:
            return (dates[0], self.end_date)
        if dates[0] >= self.end_date and dates[-1] > self.start_date:
            return (self.start_date, dates[-1])


if __name__ == '__main__':
    dataset = NormalDataset(None, None, "offline", None)

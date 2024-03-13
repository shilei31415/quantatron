# --*-- conding:utf-8 --*--
# @Time : 2023/12/18 下午3:13
# @Author : Shi Lei
# @Email : 2055833480@qq.com
# @File : configs.py
# @Software : PyCharm

from quantatron.config import CfgNode as CN
from copy import deepcopy

def add_config(cfg):
    _C = cfg

    # train dataset
    _C.TRAIN_DATASET = CN()
    _C.TRAIN_DATASET.NAME = ""
    _C.TRAIN_DATASET.START_DATE = ""
    _C.TRAIN_DATASET.END_DATE = ""
    _C.TRAIN_DATASET.MODE = ""
    _C.TRAIN_DATASET.DIR = ""
    _C.TRAIN_DATASET.STOCK_CODES = ("", "")
    _C.TRAIN_DATASET.LENGTH = 0

    # test dataset
    _C.TEST_DATASET = deepcopy(_C.TRAIN_DATASET)


    _C.MODEL = CN()
    _C.MODEL.META_ARCH = ""
    _C.MODEL.DEVICE = "cuda"
    _C.MODEL.LENGTH = 0
    _C.MODEL.COLUMN = 0
    _C.MODEL.HIDDEN_DIM = 0
    _C.MODEL.LAYER = 0
    _C.MODEL.OUTPUT_DIM = 0








# --*-- conding:utf-8 --*--
# @Time : 2023/12/18 下午3:17
# @Author : Shi Lei
# @Email : 2055833480@qq.com
# @File : test_func.py
# @Software : PyCharm

# 测试每个编写好的函数

import sys

# linux
sys.path.append("/home/shilei/Desktop/Quantatron")

# windows
sys.path.append("E:\量化交易\quantatron")

from quantatron.config import get_cfg
from quantatron.engine import default_argument_parser
from configs.config import add_config
from quantatron.utils import get_sha
from quantatron.utils.collect_env import collect_env_info
from quantatron.data import build_dataset
from quantatron.modeling.meta_arch import build_model

from quantatron.data.ts_data import TSDailyDataset, TSIncomeDataset

import tushare as ts
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args)

    ts.set_token(cfg.TS_TOKEN)
    return cfg


def main(args=None):
    cfg = setup(args)

    train_dataset, test_dataset = build_dataset(cfg)

    batch = 64

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    model = build_model(cfg)

    optimizer = optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    criterion = nn.MSELoss()

    for epoch in range(500):
        running_loss = 0.0
        for data, y in train_dataloader:
            data, y = data.to("cuda"), y.to("cuda")
            optimizer.zero_grad()
            outputs = model(data.unsqueeze(1))
            loss = criterion(outputs, y)/batch

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss

        running_loss = 0
        for data, y in test_dataloader:
            data, y = data.to("cuda"), y.to("cuda")
            outputs = model(data.unsqueeze(1))
            loss = criterion(outputs, y)

            running_loss += loss.item()
        test_loss = running_loss

        print(f"epoch: {epoch}, \ttrain loss: {train_loss:.2f}, \ttest loss: {test_loss:.2f}")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    print(get_sha())
    print(collect_env_info())
    main(args)

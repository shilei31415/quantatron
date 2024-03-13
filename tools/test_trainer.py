# --*-- conding:utf-8 --*--
# @Time : 2024/3/5 下午2:22
# @Author : Shi Lei
# @Email : 2055833480@qq.com
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

from quantatron.engine import DefaultTrainer
import tushare as ts


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
    trainer = DefaultTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    print(get_sha())
    print(collect_env_info())
    main(args)
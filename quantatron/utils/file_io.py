# --*-- conding:utf-8 --*--
# @Time : 2023/12/18 下午3:00
# @Author : Shi Lei
# @Email : 2055833480@qq.com
# @File : file_io.py
# @Software : PyCharm

# most code copy from detectron2
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
from iopath.common.file_io import PathManager as PathManagerBase

__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()

PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())

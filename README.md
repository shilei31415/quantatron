# Quantatron
## 项目介绍
参考
[detectron2](https://github.com/facebookresearch/detectron2)
编写一个使用神经网络实现的量化交易框架。

Most of code copy from 
[detectron2](https://github.com/facebookresearch/detectron2).

包含基本的数据集、日志等功能。

导入该框架，快速实现神经网络的编写

##安装
python setup.py install


## 功能模块
- **config/**: 基础的config文件

- **dataset/**： 基础的dataset使用实例

- **quantatron/**:   

  - **checkpoint/**: 断点续训，early stop，模型保存等功能。  

  - **configs/**: 默认参数，参数加载，参数修改，参数保存 等功能的实现。

  - **data/**: 使用现有量化交易api获取数据，数据集、数据增强等功能
  
  - **engine/**: 如何训练模型
  
  - **evaluation/**: 量化交易测试方法
  
  - **modeling/**: 深度学习模块的基本实现
  
  - **utils/**: 其他常用函数

- **scripts/**: 主要脚本

- **tools/**: main函数实现，可视化，测试 等功能

##注意事项：
全篇日期统一为年-月-日 月和日保留两位
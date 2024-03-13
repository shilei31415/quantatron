# --*-- conding:utf-8 --*--
# @Time : 2024/3/13 上午10:03
# @Author : Shi Lei
# @Email : 2055833480@qq.com
from .evaluator import DatasetEvaluator
import numpy as np
from scipy.stats import pearsonr

class BaseStockEvaluator(DatasetEvaluator):
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.ground_truths = []

    def process(self, inputs, outputs):
        gt = inputs["gt"].detach().squeeze().cpu().numpy()
        pred = outputs.detach().cpu().numpy()
        self.predictions.append(pred)
        self.ground_truths.append(gt)

    def evaluate(self):
        result = {}

        pred = np.concatenate(self.predictions, axis=0)
        gt = np.concatenate(self.ground_truths, axis=0)
        length, width = gt.shape[0], gt.shape[1]
        mse = np.mean((pred-gt)**2)/length
        result["MSE_loss"] = mse

        for i in range(width):
            coor, _ = pearsonr(pred[:, i], gt[:, i])
            result[f"correlation_{i}"] = coor
        return result
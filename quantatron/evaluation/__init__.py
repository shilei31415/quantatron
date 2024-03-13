from .evaluator import DatasetEvaluator, DatasetEvaluators, inference_context, inference_on_dataset
from .testing import print_csv_format, verify_results
from .stock_evaluator import BaseStockEvaluator

__all__ = [k for k in globals().keys() if not k.startswith("_")]

def build_evaluator(cfg, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    return BaseStockEvaluator()
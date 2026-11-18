"""
experiments/scanner.py
======================
通用参数扫描框架。

提供 ``run_scan`` 函数，接受一个"单次模拟"的可调用对象和参数列表，
自动循环执行并将结果汇总为 DataFrame，支持追加写入 CSV 文件。

用法示例
--------
::

    from experiments.scanner import run_scan

    def single_run(d_det):
        # ... 构建 emitter、Bragg、执行模拟 ...
        return {"detector_distance": d_det, "reflected_photons": count}

    df = run_scan(
        param_name="detector_distance",
        param_values=np.linspace(80, 200, 13),
        run_func=single_run,
        csv_path="results/scan_distance.csv",
    )
"""

import os
import pandas as pd
from typing import Callable, Iterable, Any


def run_scan(param_name: str,
             param_values: Iterable,
             run_func: Callable[[Any], dict],
             csv_path: str = None,
             verbose: bool = True) -> pd.DataFrame:
    """
    对一个参数进行遍历扫描，收集每次模拟的结果。

    参数
    ----
    param_name : str
        扫描参数的名称（仅用于日志输出）。
    param_values : iterable
        参数取值列表。
    run_func : callable
        单次模拟函数，接受一个参数值，返回一个 dict（键为列名，值为数值）。
    csv_path : str, optional
        结果 CSV 文件路径。若文件已存在，则以追加模式写入；否则新建。
        若为 None，则不写入文件。
    verbose : bool
        是否打印每次迭代的进度信息，默认 True。

    返回
    ----
    df : pd.DataFrame
        本次扫描的全部结果。
    """
    records = []
    for val in param_values:
        if verbose:
            print(f"=== Running {param_name} = {val} ===")
        result = run_func(val)
        records.append(result)

    df = pd.DataFrame(records)

    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        file_exists = os.path.isfile(csv_path)
        df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
        print(f"[Scanner] 结果已{'追加' if file_exists else '写入'} → {csv_path}")

    return df

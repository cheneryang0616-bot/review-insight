# -*- coding: utf-8 -*-
"""只读机器级预测；不加载权重，不导入 torch，不向 LLM 暴露 npz。"""
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

BASE_COLUMNS = ("unit", "y_true", "pred", "sigma", "member_std", "end_cycle")


def _frame(values, **attrs):
    df = pd.DataFrame(values)
    missing = set(BASE_COLUMNS) - set(df)
    if missing:
        raise ValueError(f"PHM 预测缺少列：{sorted(missing)}")
    if df.empty:
        raise ValueError("PHM 预测为空")
    for col in BASE_COLUMNS:
        # float32 先无损提升为 float64，确保 CSV 导出保留原始预测值，
        # 避免 pandas 的 float32 最短表示在样例中引入二次量化。
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"PHM {col} 含非有限数值")
    for col in ("unit", "end_cycle"):
        if (df[col] <= 0).any() or (df[col] != np.floor(df[col])).any():
            raise ValueError(f"PHM {col} 必须为正整数")
        df[col] = df[col].astype(int)
    if df["unit"].duplicated().any():
        raise ValueError("PHM unit 重复，预测无法按机器对齐")
    if (df[["sigma", "member_std", "y_true"]] < 0).any().any():
        raise ValueError("PHM sigma、member_std 和 y_true 不得为负")
    df = df.sort_values("unit").reset_index(drop=True)
    if not np.array_equal(df["unit"], np.arange(1, len(df) + 1)):
        raise ValueError("PHM unit 必须覆盖 1..N")
    df["region"] = np.select(
        [df["pred"] <= config.SERVICE_NEAR, df["pred"] <= config.SERVICE_MID],
        ["near", "mid"], default="far")
    df["lower"] = df["pred"] - config.INTERVAL_K * df["sigma"]
    df["upper"] = df["pred"] + config.INTERVAL_K * df["sigma"]
    df.attrs.update(attrs)
    return df


def load_sample(path=None):
    """独立 clone 使用导出的真实机器摘要，不需要另一仓库。"""
    path = Path(path) if path is not None else config.SAMPLE_PATH
    raw = pd.read_csv(path)
    for key, expected in (("dataset", config.DATASET), ("run", config.RUN)):
        if key not in raw or not raw[key].eq(expected).all():
            raise ValueError(f"PHM 样例 {key} 与默认预测来源不一致")
    return _frame(raw.loc[:, list(BASE_COLUMNS)], dataset=config.DATASET,
                  run=config.RUN, source="sample", path=str(path), root=None)


def load(root=None):
    """RI_PHM_ROOT 指向外部资产。显式配置出错时不静默回退。"""
    env_root = os.environ.get("RI_PHM_ROOT")
    explicit = root is not None or env_root is not None
    root = Path(root if root is not None else env_root or config.DEFAULT_ROOT).expanduser()
    if not root.exists() and not explicit:
        return load_sample()
    path = root / "artifacts" / config.RUN / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"PHM 预测不存在：{path}；请检查 RI_PHM_ROOT")
    keys = ("u_test", "y_test", "pred_ensemble", "sigma", "end_cycle")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys + ("pred_members",)) - set(archive.files)
        if missing:
            raise ValueError(f"PHM npz 缺少字段：{sorted(missing)}")
        arrays = {key: archive[key].copy() for key in keys}
        if any(value.ndim != 1 for value in arrays.values()):
            raise ValueError("PHM npz 机器级字段长度或维度不一致")
        n = len(arrays["u_test"])
        if any(len(value) != n for value in arrays.values()):
            raise ValueError("PHM npz 机器级字段长度或维度不一致")
        members = np.asarray(archive["pred_members"], dtype=float)
        if members.ndim != 2 or members.shape[0] == 0 or members.shape[1] != n:
            raise ValueError("PHM pred_members 必须为 (成员数, 机器数)")
        if not np.isfinite(members).all():
            raise ValueError("PHM pred_members 含非有限数值")
    return _frame({
        "unit": arrays["u_test"], "y_true": arrays["y_test"],
        "pred": arrays["pred_ensemble"], "sigma": arrays["sigma"],
        "member_std": members.std(axis=0), "end_cycle": arrays["end_cycle"],
    }, dataset=config.DATASET, run=config.RUN, source="full", path=str(path), root=str(root))


@lru_cache(maxsize=4)
def _metrics(path):
    # 仅加载纯 numpy 的 metrics.py，避免 cmapss 包初始化和 torch 依赖。
    spec = importlib.util.spec_from_file_location("_phm_cmapss_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coverage(df):
    """完整资产复用 cmapss.metrics.coverage；样例用同一闭区间公式。"""
    root = df.attrs.get("root")
    if root:
        metrics_path = Path(root) / "src" / "cmapss" / "metrics.py"
        if metrics_path.is_file():
            return _metrics(str(metrics_path)).coverage(
                df["pred"], df["y_true"], df["sigma"], k=config.INTERVAL_K)
    pred, y, sigma = (df[key].to_numpy(dtype=float) for key in ("pred", "y_true", "sigma"))
    return float(np.mean((y >= pred - config.INTERVAL_K * sigma)
                         & (y <= pred + config.INTERVAL_K * sigma)))


def scope_desc(df=None, **scope):
    df = load() if df is None else df
    source = "真实预测资产" if df.attrs["source"] == "full" else "仓库自带的真实预测摘要样例"
    return (f"C-MAPSS 仿真数据 {df.attrs['dataset']}；{len(df)} 台测试发动机；"
            f"{source}；运行 {df.attrs['run']}。RUL 单位为运行循环数。")

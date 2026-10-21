# -*- coding: utf-8 -*-
"""只读机器级预测；不加载权重，不导入 torch，不向 LLM 暴露 npz。"""
import datetime
import hashlib
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

BASE_COLUMNS = ("unit", "y_true", "pred", "sigma", "member_std", "end_cycle")
DATASETS = ("FD001", "FD002", "FD003", "FD004")
SENSOR_COLUMNS = tuple(f"sensor_{i}" for i in range(1, 22))
PANEL_COLUMNS = ("unit", "cycle", "setting_1", "setting_2", "setting_3") + SENSOR_COLUMNS
COMPARABLE_MODELS = ("cnn_bilstm_attn", "gbdt", "patch_transformer",
                     "legacy_cnn_bilstm", "cnn_transformer")


def resolve_dataset(dataset=None):
    dataset = config.DATASET if dataset is None else str(dataset).upper()
    if dataset not in DATASETS:
        raise ValueError(f"PHM 数据集必须是 {', '.join(DATASETS)}")
    return dataset


def resolve_root(root=None):
    """解析资产路径；显式 root 优先于环境变量，永不替换错误配置。"""
    value = root if root is not None else os.environ.get("RI_PHM_ROOT", config.DEFAULT_ROOT)
    return Path(value).expanduser().resolve()


def run_name(dataset=None, model="cnn_bilstm_attn"):
    return f"{resolve_dataset(dataset)}_{model}_w30_mrul125_final"


def asset_context(df, dataset=None):
    """供依赖原始资产的工具使用，样例不能借默认路径读取另一份资产。"""
    dataset = resolve_dataset(dataset if dataset is not None else df.attrs.get("dataset"))
    root = df.attrs.get("root")
    if df.attrs.get("source") != "full" or root is None:
        raise FileNotFoundError("PHM 摘要样例不含原始序列、训练统计和模型报告")
    return {"root": resolve_root(root), "dataset": dataset}


@lru_cache(maxsize=16)
def _sha256_cached(path_str, mtime_ns, size):
    """整个文件算一次 sha256；缓存键用 mtime+size，文件一变自动重算。

    只读预测文件只有几 KB（FD001 7.7K / FD004 17K），全量哈希成本可忽略，
    比"只哈希前 4MB"更不容易出现"文件变了指纹没变"的假相同。
    """
    return hashlib.sha256(Path(path_str).read_bytes()).hexdigest()


def fingerprint(path):
    """预测资产的指纹：来源类型 + sha256 前 8 位 + 大小 + mtime。

    为什么要这个：报告里写着"运行 FD001_xxx_final"，但没人知道那份 npz 是不是被替换过。
    指纹让"这份结论来自哪个文件、哪一版"变成可核对的事实，而不是凭记忆。
    """
    path = Path(path)
    stat = path.stat()
    sha = _sha256_cached(str(path), int(stat.st_mtime_ns), int(stat.st_size))
    kind = "npz" if path.suffix.lower() == ".npz" else "sample"
    return {
        "asset_id": f"{kind}:{sha[:8]}",
        "asset_sha256": sha,
        "asset_kind": kind,
        "asset_bytes": int(stat.st_size),
        "asset_mtime_ns": int(stat.st_mtime_ns),
    }


def asset_summary(df):
    """给工具的 facts / 展示层用的指纹摘要（含人类可读时间）。"""
    ns = df.attrs.get("asset_mtime_ns")
    iso = ""
    if ns:
        try:
            iso = datetime.datetime.fromtimestamp(int(ns) / 1e9).strftime("%Y-%m-%d %H:%M")
        except (OverflowError, OSError, ValueError):
            iso = ""
    return {
        "asset_id": str(df.attrs.get("asset_id") or "unknown"),
        "asset_sha256": str(df.attrs.get("asset_sha256") or ""),
        "asset_kind": str(df.attrs.get("asset_kind") or ""),
        "asset_bytes": int(df.attrs.get("asset_bytes") or 0),
        "asset_mtime_ns": int(ns or 0),
        "asset_mtime_iso": iso,
    }


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
                  run=config.RUN, source="sample", path=str(path), root=None,
                  **fingerprint(path))


def load(root=None, dataset=None):
    """RI_PHM_ROOT 指向外部资产。显式配置出错时不静默回退。"""
    env_root = os.environ.get("RI_PHM_ROOT")
    explicit = root is not None or env_root is not None
    dataset = resolve_dataset(dataset)
    root = resolve_root(root)
    if not root.exists() and not explicit:
        if dataset == config.DATASET:
            return load_sample()
        raise FileNotFoundError(f"PHM {dataset} 的资产不存在，仓库样例仅支持 {config.DATASET}")
    run = run_name(dataset)
    path = root / "artifacts" / run / "predictions.npz"
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
    }, dataset=dataset, run=run, source="full", path=str(path), root=str(root),
        **fingerprint(path))


@lru_cache(maxsize=16)
def _read_panel_cached(path, mtime_ns, size):
    # mtime/size 参与缓存键，更新原始文件后不复用旧内容。
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] != len(PANEL_COLUMNS) or raw.empty:
        raise ValueError(f"PHM 原始序列必须是非空的 26 列：{path}")
    raw.columns = PANEL_COLUMNS
    raw = raw.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError(f"PHM 原始序列含非有限值：{path}")
    for key in ("unit", "cycle"):
        if (raw[key] <= 0).any() or (raw[key] != np.floor(raw[key])).any():
            raise ValueError(f"PHM 原始序列 {key} 必须为正整数：{path}")
        raw[key] = raw[key].astype(int)
    if raw.duplicated(["unit", "cycle"]).any():
        raise ValueError(f"PHM 原始序列的 unit/cycle 重复：{path}")
    return raw.sort_values(["unit", "cycle"]).reset_index(drop=True)


def _raw_panel(root, dataset, split):
    path = resolve_root(root) / "data" / f"{split}_{resolve_dataset(dataset)}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"PHM 原始序列不存在：{path}")
    stat = path.stat()
    return _read_panel_cached(str(path), stat.st_mtime_ns, stat.st_size)


def panel(unit, dataset=None, root=None):
    """返回单机原始测试序列的副本；缓存内容不受调用者修改影响。"""
    if isinstance(unit, bool) or not np.isfinite(float(unit)) or float(unit) != int(unit):
        raise ValueError("PHM unit 必须为正整数")
    raw = _raw_panel(root, dataset, "test")
    result = raw.loc[raw["unit"].eq(int(unit))].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError(f"PHM 原始序列中不存在机器 {unit}")
    result.attrs.update(dataset=resolve_dataset(dataset), source="raw_test",
                        root=str(resolve_root(root)))
    return result


def sensor_stats(dataset=None, root=None):
    """训练集传感器总体统计量（ddof=0），不使用测试机或未来 RUL。"""
    dataset = resolve_dataset(dataset)
    raw = _raw_panel(root, dataset, "train")
    sensors = raw.loc[:, list(SENSOR_COLUMNS)]
    std = sensors.std(ddof=0)
    # 常量通道在浮点均值运算中可能出现极小的伪标准差。
    std.loc[sensors.nunique().eq(1)] = 0.0
    first = raw.groupby("unit", sort=False).head(30).groupby("unit")[list(SENSOR_COLUMNS)].mean()
    last = raw.groupby("unit", sort=False).tail(30).groupby("unit")[list(SENSOR_COLUMNS)].mean()
    direction = np.sign((last - first).mean()).replace(0, 1)
    result = pd.DataFrame({"mean": sensors.mean(), "std": std, "direction": direction})
    multi = dataset in ("FD002", "FD004")
    result.attrs.update(
        dataset=dataset, root=str(resolve_root(root)), source="raw_train",
        is_multi_condition=multi, normalization="global_training_statistics",
        note=(("使用全训练集均值与标准差的通道偏离代理；多工况未按工况分组标准化。" if multi
               else "使用全训练集均值与标准差的通道偏离代理。")
              + "健康指数方向取训练整机末窗口减首窗口均值的平均趋势，属于代理指标。"))
    return result


def runs(dataset=None, root=None):
    """只比较 summary 中同协议的五个正式模型，并核对各自原始报告。"""
    dataset, root = resolve_dataset(dataset), resolve_root(root)
    summary_path = root / "artifacts" / f"summary_{dataset}.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"PHM 模型汇总不存在：{summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = []
    for model in COMPARABLE_MODELS:
        if model not in summary:
            raise ValueError(f"PHM 汇总缺少正式模型 {model}：{summary_path}")
        run = (f"{dataset}_cnn_bilstm_attn_w30_mrul125_gbdt" if model == "gbdt"
               else run_name(dataset, model))
        path = root / "artifacts" / run / "report.json"
        if not path.is_file():
            raise FileNotFoundError(f"PHM 正式模型报告不存在：{path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("run_name") != run:
            raise ValueError(f"PHM 报告运行名与目录不一致：{path}")
        protocol = report.get("config", {}).get("data", {})
        if any(protocol.get(key) != value for key, value in (
                ("dataset", dataset), ("window", 30), ("max_rul", 125), ("rul_mode", "clip"))):
            raise ValueError(f"PHM 报告与正式比较协议不一致：{path}")
        metrics = report.get("ensemble", {})
        record = {"model": model, "source": str(path), "summary_source": str(summary_path)}
        for key in ("rmse", "nasa_score"):
            value, expected = float(metrics[key]), float(summary[model][key])
            if (not np.isfinite(value) or value < 0
                    or not np.isclose(value, expected, rtol=1e-10, atol=1e-12)):
                raise ValueError(f"PHM {model} 的 {key} 与汇总不一致：{path}")
            record[key] = value
        if metrics.get("n") != summary[model].get("n") or not metrics.get("n"):
            raise ValueError(f"PHM {model} 的测试样本量与汇总不一致：{path}")
        record["n"] = int(metrics["n"])
        result.append(record)
    return result


def pipeline_check(df):
    """将预测按 unit 与原始 RUL/测试终点核对，缺原始资产时明确不可验证。"""
    facts = {"pipeline_ok": False, "u_count": len(df["unit"]), "y_count": len(df["y_true"]),
             "rul_count": -1, "panel_unit_count": -1, "alignment_checked": False,
             "labels_match": False, "end_cycles_match": False, "unit_alignment_ok": False}
    try:
        context = asset_context(df)
        raw = _raw_panel(context["root"], context["dataset"], "test")
        path = context["root"] / "data" / f"RUL_{context['dataset']}.txt"
        if not path.is_file():
            return facts
        rul = pd.read_csv(path, sep=r"\s+", header=None)
        if rul.shape[1] != 1 or rul.empty:
            return facts
        y = pd.to_numeric(rul.iloc[:, 0], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y < 0).any():
            return facts
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        return facts
    facts["rul_count"], facts["panel_unit_count"] = len(y), int(raw["unit"].nunique())
    units = df["unit"].to_numpy(dtype=float)
    valid_units = (np.isfinite(units).all() and np.equal(units, np.floor(units)).all()
                   and len(set(units)) == len(units)
                   and np.array_equal(np.sort(units), np.arange(1, len(y) + 1)))
    facts["alignment_checked"] = True
    facts["unit_alignment_ok"] = bool(
        valid_units and set(units) == set(raw["unit"].unique()))
    if facts["unit_alignment_ok"]:
        expected = np.clip(y[units.astype(int) - 1], 0, 125)
        facts["labels_match"] = bool(np.allclose(df["y_true"], expected, rtol=0, atol=1e-6))
        end_cycles = raw.groupby("unit")["cycle"].max().reindex(units.astype(int))
        facts["end_cycles_match"] = bool(np.array_equal(df["end_cycle"], end_cycles))
    facts["pipeline_ok"] = bool(
        facts["u_count"] == facts["y_count"] == facts["rul_count"] == facts["panel_unit_count"]
        and facts["unit_alignment_ok"] and facts["labels_match"] and facts["end_cycles_match"])
    return facts


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
    if df is None:
        df = load(dataset=scope.get("dataset"))
    requested = resolve_dataset(scope.get("dataset") or df.attrs.get("dataset"))
    if requested != df.attrs.get("dataset"):
        try:
            df = load(**asset_context(df, dataset=requested))
        except FileNotFoundError:
            return (f"C-MAPSS 仿真数据 {requested} 当前不可用；当前来源未提供该数据集资产。"
                    "RUL 单位为运行循环数。")
    source = "真实预测资产" if df.attrs["source"] == "full" else "仓库自带的真实预测摘要样例"
    info = asset_summary(df)
    stamp = f"，{info['asset_mtime_iso']}" if info["asset_mtime_iso"] else ""
    return (f"C-MAPSS 仿真数据 {df.attrs['dataset']}；{len(df)} 台测试发动机；"
            f"{source}；运行 {df.attrs['run']}；预测资产 {info['asset_id']}{stamp}。"
            f"RUL 单位为运行循环数。")

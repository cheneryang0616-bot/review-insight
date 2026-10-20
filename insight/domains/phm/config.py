# -*- coding: utf-8 -*-
"""PHM Phase 0：预测来源、区间口径与显式假设。"""
from pathlib import Path

DEFAULT_ROOT = Path("/Users/zhuan/c-mapss_train")
DATASET = "FD001"
# 验收中的 56% 属于此集成；patch_transformer 的覆盖率是 50%。
RUN = "FD001_cnn_bilstm_attn_w30_mrul125_final"
SAMPLE_PATH = Path(__file__).resolve().parents[3] / "data" / "sample_phm_units.csv"
NOMINAL = 0.95
INTERVAL_K = 1.96
COVERAGE_TOLERANCE = 0.10
MIN_COVERAGE = NOMINAL - COVERAGE_TOLERANCE
SERVICE_NEAR = 30
SERVICE_MID = 60

# 示意值；Phase 0 不执行成本优化或排期，也不把循环数换算成天。
C_FN = 10.0
C_FP = 1.0
LEAD_TIME_DAYS = 14

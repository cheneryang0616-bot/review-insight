# -*- coding: utf-8 -*-
"""PHM：预测来源、决策口径与可覆盖的示意假设。"""
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

# 示意值，绝非真实产线参数；工具必须随结果披露。
C_FN = 10.0
C_FP = 1.0
LEAD_TIME_DAYS = 14
WINDOW_DAYS = 14
CYCLES_PER_DAY = 1.0
RISK_HORIZON_CYCLES = 30.0
DIAGNOSTIC_WINDOW = 30
SMOOTHING_WINDOW = 5
HEALTH_SENSORS = ("sensor_2", "sensor_7", "sensor_11", "sensor_12", "sensor_15")
ACCELERATION_MARGIN = 0.20
CHOSEN_MODEL = "cnn_bilstm_attn"

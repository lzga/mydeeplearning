# Small-object-aware NWD 修改说明

本版本删除了此前全局 NWD 的配置与入口，改为 small-object-aware NWD：

- 删除旧配置：`configs/rtdetr/baseline_nwd.yml`、`configs/rtdetr/fb_c5_assa_nwd.yml`
- 新增配置：
  - `configs/rtdetr/baseline_soa_nwd.yml`
  - `configs/rtdetr/fb_c5_assa_soa_nwd.yml`
- 修改 `src/zoo/rtdetr/rtdetr_criterion.py`
  - 删除旧的 `use_nwd`、`nwd_weight`、`loss_nwd`、`_compute_nwd_loss`
  - 新增 `use_small_object_aware_nwd`、`loss_soa_nwd`、`_compute_small_object_aware_nwd_loss`
  - 默认仅对最终 decoder 输出启用 NWD，不对 aux/dn 分支启用
- 修改 `configs/rtdetr/include/rtdetr_r50vd.yml`
  - `weight_dict` 中使用 `loss_soa_nwd`
  - 添加 small-object-aware NWD 默认配置，默认关闭
- 修改 `mytools/run_nwd_experiments.py`
  - 串行运行两组实验：`baseline_soa_nwd` 和 `fb_c5_assa_soa_nwd`
  - 两组全部正常结束后自动关机
  - 如果有实验失败，默认不自动关机；如需失败后也关机，设置 `SHUTDOWN_ON_ERROR = True`

当前默认权重策略（mAP50-oriented）：

```text
s = sqrt(w_gt * h_gt) * 640
s < 16px        -> NWD weight = 0.15
16px <= s < 32px -> NWD weight = 0.05
s >= 32px       -> NWD weight = 0.00
```


## mAP50-oriented parameter update

本版本将 small-object-aware NWD 调整为更保守、更偏向 mAP50 的设置：

```text
s = sqrt(w_gt * h_gt) * 640
s < 16px        -> NWD weight = 0.15
16px <= s < 32px -> NWD weight = 0.05
s >= 32px       -> NWD weight = 0.00
```

调整原因：前一版 `0.25 / 0.10 / 0.00` 更偏向 AP_small 与 mAP50-95；若目标是提升 mAP50，需要减弱 NWD 对回归分支的约束，优先保护召回、粗定位命中和分类置信度排序。

输出目录已改为：

- `output/baseline_soa_nwd_map50`
- `output/fb_c5_assa_soa_nwd_map50`

运行脚本 `mytools/run_nwd_experiments.py` 会串行运行以上两组实验，全部成功后自动关机。

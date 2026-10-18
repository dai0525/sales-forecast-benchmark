| 模型 | 平均 WAPE | 总量 WAPE |
|---|---:|---:|
| 联合状态 Fourier | 7.313% | 5.476% |
| UCM52（MLE） | 7.840% | 5.565% |
| LightGBM 全训练行 | 7.960% | 5.713% |
| BSTS MCMC†（有限预算，收敛受限） | 7.981% | 5.734% |
| PatchTST | 8.132% | 5.989% |
| FG backfitting 原型 | 8.706% | 5.980% |
| Naive | 9.001% | 6.340% |
| Pooled residual MLP (NHITS-inspired) | 9.360% | 6.713% |
| TFT-style median model | 10.463% | 8.311% |
| MovingAvg8 | 10.836% | 7.840% |
| N-BEATS-style (no exogenous inputs) | 11.864% | 8.335% |
| Direct Gaussian LSTM (not DeepAR) | 13.600% | 10.422% |
| SeasonalNaive52 | 15.553% | 13.249% |

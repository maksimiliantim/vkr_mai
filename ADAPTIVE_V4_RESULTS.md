# AdaptiveLossWeights v4 results

Версия `v4` сохраняет постановку из презентации: Fourier-признаки, MLP 6×160,
постоянные параметры ODE `beta`, `gamma`, `kappa`, `delta`, те же режимы
`reconstruction` и `holdout`. Добавлено только обучение весов функции потерь
`AdaptiveLossWeights`; веса обучаются медленнее основной модели через
`loss_weight_lr_scale=0.1`.

| Папка | Train R² | Test R² | lambda_data | lambda_ode |
|---|---:|---:|---:|---:|
| `final_2017_reconstruction` | 0.8772 | — | 0.8244 | 3.484e-05 |
| `final_2017_holdout` | 0.8756 | 0.8736 | 0.8268 | 3.504e-05 |
| `final_2018_reconstruction` | 0.9182 | — | 0.9832 | 4.454e-06 |
| `final_2018_holdout` | 0.9447 | 0.8615 | 0.9847 | 4.560e-06 |

Для воспроизведения запустить: `bash run_all_final_adaptive.sh`

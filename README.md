# PINN-модель динамики киберугроз

Репозиторий содержит самодостаточный вычислительный конвейер для экспериментов
ВКР: подготовку агрегированного временного ряда из сырых CSV, обучение
регуляризованной PINN-модели, сохранение предсказаний, параметров и метрик.

Модель не пытается воспроизвести каждый пик временного ряда через пооконный
обучаемый форсинг. Она рассматривается как инструмент регуляризованной
реконструкции базовой динамики, идентификации параметров и анализа невязки ОДУ.

## Структура

- `prepare_cic2017.py` - подготовка CIC-IDS2017 из сырых CSV в `.npz`;
- `prepare_cse2018.py` - подготовка CSE-CIC-IDS2018 из сырых CSV в `.npz`;
- `data_preprocessing.py` - очистка, суррогатная временная ось, агрегация и нормировка CIC-IDS2017;
- `data_preprocessing_cse2018.py` - обработка Timestamp, удаление календарных разрывов, агрегация и нормировка CSE-CIC-IDS2018;
- `run_experiment.py` - обучение PINN и сохранение `summary.json`, `predictions.csv`, `model.pt`;
- `final_*` - сохраненные финальные результаты для диплома.

## Зависимости

```bash
python -m pip install -r requirements.txt
```

Для обучения нужен PyTorch. Если он уже установлен в используемом окружении,
достаточно запускать скрипты этим же интерпретатором Python.

## Полный запуск от сырых CSV

### CIC-IDS2017

```bash
python prepare_cic2017.py \
  --data_dir /path/to/MachineLearningCVE \
  --output data/cic2017_preprocessed.npz \
  --delta_t 60

python run_experiment.py \
  --preprocessed data/cic2017_preprocessed.npz \
  --dataset CIC-IDS2017 \
  --mode reconstruction \
  --output_dir final_2017_reconstruction \
  --hidden_dim 160 \
  --n_layers 6 \
  --n_harmonics 16 \
  --n_state_harmonics 48 \
  --epochs 4000 \
  --min_epochs 1200 \
  --patience 700 \
  --lambda_ode 5e-5 \
  --lambda_ic 0.1 \
  --lambda_forcing 1e-4 \
  --lambda_smooth 1e-3 \
  --lambda_d_smooth 5e-6 \
  --peak_weight 5

python run_experiment.py \
  --preprocessed data/cic2017_preprocessed.npz \
  --dataset CIC-IDS2017 \
  --mode holdout \
  --output_dir final_2017_holdout \
  --hidden_dim 160 \
  --n_layers 6 \
  --n_harmonics 16 \
  --n_state_harmonics 48 \
  --epochs 4000 \
  --min_epochs 1200 \
  --patience 700 \
  --lambda_ode 5e-5 \
  --lambda_ic 0.1 \
  --lambda_forcing 1e-4 \
  --lambda_smooth 1e-3 \
  --lambda_d_smooth 5e-6 \
  --peak_weight 5
```

### CSE-CIC-IDS2018

```bash
python prepare_cse2018.py \
  --data_dir /path/to/CSE-CIC-IDS2018 \
  --output_dir data/cse2018_preprocessed \
  --n_windows 2400

python run_experiment.py \
  --preprocessed data/cse2018_preprocessed/cse2018_preprocessed.npz \
  --dataset CSE-CIC-IDS2018 \
  --mode reconstruction \
  --output_dir final_2018_reconstruction \
  --hidden_dim 160 \
  --n_layers 6 \
  --n_harmonics 32 \
  --n_state_harmonics 128 \
  --epochs 5000 \
  --min_epochs 1500 \
  --patience 900 \
  --lambda_ode 5e-6 \
  --lambda_ic 0.1 \
  --lambda_forcing 5e-6 \
  --lambda_smooth 5e-5 \
  --lambda_d_smooth 0 \
  --peak_weight 300

python run_experiment.py \
  --preprocessed data/cse2018_preprocessed/cse2018_preprocessed.npz \
  --dataset CSE-CIC-IDS2018 \
  --mode holdout \
  --output_dir final_2018_holdout \
  --hidden_dim 160 \
  --n_layers 6 \
  --n_harmonics 32 \
  --n_state_harmonics 128 \
  --epochs 5000 \
  --min_epochs 1500 \
  --patience 900 \
  --lambda_ode 5e-6 \
  --lambda_ic 0.1 \
  --lambda_forcing 5e-6 \
  --lambda_smooth 5e-5 \
  --lambda_d_smooth 0 \
  --peak_weight 300
```

## Ключевые отличия от старого кода

- вход нейросети состояния: нормированное время `t` и Fourier-признаки времени;
- внешний форсинг `u(t)` задается гладкой низкочастотной функцией, а не отдельным параметром на каждое временное окно;
- физический штраф ОДУ используется как мягкая регуляризация, а не как механизм подгонки каждого локального всплеска;
- обучение выполняется единой оптимизацией Adam с ранней остановкой;
- отдельно считаются метрики для `reconstruction`, `holdout` и `forecast`;
- `forecast` оставлен как диагностический режим, но основной вывод диплома строится по `reconstruction` и `holdout`.

## Интерпретация

`R²` относится только к наблюдаемой компоненте `D(t)`. Для латентного риска
`I(t)`, внешнего форсинга `u(t)`, параметров `β`, `γ`, `κ`, `δ` и невязки ОДУ
нет прямых эталонных наблюдений, поэтому для них `R²` не рассчитывается.

## Финальная конфигурация

| Параметр | CIC-IDS2017 | CSE-CIC-IDS2018 |
|---|---:|---:|
| `hidden_dim` | 160 | 160 |
| `n_layers` | 6 | 6 |
| `n_harmonics` | 16 | 32 |
| `n_state_harmonics` | 48 | 128 |
| `epochs` | 4000 | 5000 |
| `min_epochs` | 1200 | 1500 |
| `patience` | 700 | 900 |
| `lambda_ode` | `5e-5` | `5e-6` |
| `lambda_ic` | `0.1` | `0.1` |
| `lambda_forcing` | `1e-4` | `5e-6` |
| `lambda_smooth` | `1e-3` | `5e-5` |
| `lambda_d_smooth` | `5e-6` | `0` |
| `peak_weight` | 5 | 300 |

## Финальные значения

| Режим | Train R² | Test R² | Комментарий |
|---|---:|---:|---|
| CIC-IDS2017 reconstruction | 0.8710 | - | восстановление всего ряда |
| CIC-IDS2017 holdout | 0.8698 | 0.8681 | каждое 5-е окно исключается из data-loss |
| CSE-CIC-IDS2018 reconstruction | 0.9302 | - | восстановление всего ряда |
| CSE-CIC-IDS2018 holdout | 0.9468 | 0.8731 | каждое 5-е окно исключается из data-loss |

Для диплома основной акцент делается на `holdout`, потому что он показывает
качество восстановления на отложенных временных окнах:

- CIC-IDS2017: `R² = 0.8681`;
- CSE-CIC-IDS2018: `R² = 0.8731`.

Основная постановка работы: реконструкция скрытой динамики, идентификация
параметров и анализ невязки ОДУ, а не прогноз будущего блока ряда.

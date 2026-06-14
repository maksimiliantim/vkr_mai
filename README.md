# PINN-модель динамики киберугроз

Репозиторий содержит воспроизводимый вычислительный конвейер для выпускной квалификационной работы по анализу динамики киберугроз с использованием физически-информированной нейронной сети, PINN.

Код решает три основные задачи:

- подготовка агрегированного временного ряда из сырых CSV-файлов сетевого трафика;
- обучение PINN-модели с ограничениями на основе системы ОДУ;
- сохранение предсказаний, метрик, параметров модели и диагностической невязки ОДУ.

Модель рассматривается не как обычная регрессионная нейросеть, а как инструмент восстановления скрытой динамики, оценки параметров и анализа участков, где наблюдаемый ряд плохо согласуется с заданной дифференциальной моделью.

## Данные

В экспериментах используются два набора данных:

- `CIC-IDS2017`;
- `CSE-CIC-IDS2018`.

Сырые CSV-файлы не включены в репозиторий из-за размера. Скрипты подготовки данных ожидают, что пользователь локально укажет путь к директории с исходными CSV-файлами.

## Структура репозитория

- `prepare_cic2017.py` - подготовка данных CIC-IDS2017;
- `prepare_cse2018.py` - подготовка данных CSE-CIC-IDS2018;
- `data_preprocessing.py` - очистка, бинаризация меток, агрегация и нормировка CIC-IDS2017;
- `data_preprocessing_cse2018.py` - обработка временных меток, агрегация и нормировка CSE-CIC-IDS2018;
- `run_experiment.py` - обучение PINN-модели и сохранение результатов;
- `requirements.txt` - список зависимостей;
- `final_2017_reconstruction/` - финальный эксперимент CIC-IDS2017 в режиме восстановления;
- `final_2017_holdout/` - финальный эксперимент CIC-IDS2017 на отложенных точках;
- `final_2018_reconstruction/` - финальный эксперимент CSE-CIC-IDS2018 в режиме восстановления;
- `final_2018_holdout/` - финальный эксперимент CSE-CIC-IDS2018 на отложенных точках.

## Установка зависимостей

```bash
python -m pip install -r requirements.txt
```

Для обучения требуется PyTorch. Если PyTorch уже установлен в используемом окружении, скрипты можно запускать этим же интерпретатором Python.

## Подготовка данных

### CIC-IDS2017

```bash
python prepare_cic2017.py \
  --data_dir /path/to/MachineLearningCVE \
  --output data/cic2017_preprocessed.npz \
  --delta_t 60
```

### CSE-CIC-IDS2018

```bash
python prepare_cse2018.py \
  --data_dir /path/to/CSE-CIC-IDS2018 \
  --output_dir data/cse2018_preprocessed \
  --n_windows 2400
```

## Запуск экспериментов

### CIC-IDS2017: reconstruction

```bash
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
```

### CIC-IDS2017: holdout

```bash
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

### CSE-CIC-IDS2018: reconstruction

```bash
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
```

### CSE-CIC-IDS2018: holdout

```bash
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

## Математическая постановка

PINN-модель восстанавливает скрытые состояния динамической системы и связывает их с наблюдаемым агрегированным рядом атаковой активности.

Используется система ОДУ с двумя состояниями:

- `$I(t)$` - скрытая интенсивность процесса;
- `$D(t)$` - наблюдаемая или накопленная компонента;
- `$u(t)$` - обучаемый внешний форсинг;
- `$\beta$`, `$\gamma$`, `$\kappa$`, `$\delta$` - оцениваемые параметры динамики.

Невязка ОДУ используется как диагностический сигнал. Она показывает, насколько восстановленная траектория согласуется с заданной динамической моделью.

## Особенности реализации

- вход модели включает нормированное время и Fourier-признаки;
- внешний форсинг `$u(t)$` задается гладкой функцией;
- физический штраф ОДУ используется как мягкая регуляризация;
- обучение выполняется через Adam с ранней остановкой;
- отдельно поддерживаются режимы `reconstruction` и `holdout`;
- режим `forecast` используется как дополнительный диагностический режим.

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

## Итоговые результаты

| Датасет и режим | Train `$R^2$` | Test `$R^2$` | Комментарий |
|---|---:|---:|---|
| CIC-IDS2017 reconstruction | 0.8772 | - | восстановление всего ряда |
| CIC-IDS2017 holdout | 0.8756 | 0.8736 | каждое 5-е окно исключается из data-loss (offset_4) |
| CSE-CIC-IDS2018 reconstruction | 0.9182 | - | восстановление всего ряда |
| CSE-CIC-IDS2018 holdout | 0.9447 | 0.8615 | каждое 5-е окно исключается из data-loss (offset_4) |

Основной акцент в работе делается на режиме `holdout`, поскольку он показывает качество восстановления на отложенных временных окнах:

- CIC-IDS2017: `$R^2 = 0.8736$` (offset_4); среднее по 5 смещениям = 0.8732 (диапазон 0.8521–0.8941);
- CSE-CIC-IDS2018: `$R^2 = 0.8615$`.

## Интерпретация метрик

Метрика `$R^2$` рассчитывается только для наблюдаемой компоненты `$D(t)$`, так как для скрытого состояния `$I(t)$`, внешнего форсинга `$u(t)$`, параметров `$\beta$`, `$\gamma$`, `$\kappa$`, `$\delta$` и невязки ОДУ нет прямых эталонных наблюдений.

Поэтому результаты модели следует интерпретировать не только как качество аппроксимации временного ряда, но и как восстановление скрытой динамики с дополнительным диагностическим сигналом.

## Выходные артефакты

После запуска эксперимента в директории результата сохраняются:

- `summary.json` - итоговые метрики, параметры и настройки запуска;
- `predictions.csv` - восстановленные значения, скрытые состояния и невязки;
- `model.pt` - сохраненные веса модели;
- PNG-графики - визуализация рядов, ошибок, параметров и диагностической невязки.

## Назначение репозитория

Репозиторий подготовлен для воспроизведения экспериментов ВКР и демонстрации программной реализации PINN-подхода к анализу динамики киберугроз.

Основная постановка работы: реконструкция скрытой динамики, идентификация параметров и анализ невязки ОДУ, а не прогнозирование будущего блока ряда.

"""
data_preprocessing.py
======================
Модуль предобработки данных датасета CIC-IDS2017 (MachineLearningCVE).

Задача: из сырых flow-записей получить временной ряд интенсивности атак D(t),
который будет использоваться как наблюдаемая величина в PINN-модели.

Схема:
  raw CSV files → parse timestamps → bin into windows Δt → compute N_k → normalize
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from export_utils import save_preprocessing_artifacts
except Exception:
    save_preprocessing_artifacts = None

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────────
# Константы
# ────────────────────────────────────────────────────────────────────────────────

# Метки атак в датасете (всё, что не BENIGN — считаем атакой)
BENIGN_LABEL = "BENIGN"

# Размер временного окна агрегирования (секунды)
DELTA_T_SECONDS = 60  # 1 минута

# Столбец с метками
LABEL_COL = " Label"

# Столбцы для временного упорядочивания (используем порядок записей как прокси)
# В датасете нет явного timestamp-столбца, поэтому используем Flow Duration
# и порядок строк как суррогатное время
FLOW_DURATION_COL = " Flow Duration"
FLOW_BYTES_COL = "Flow Bytes/s"
FLOW_PACKETS_COL = " Flow Packets/s"


# ────────────────────────────────────────────────────────────────────────────────
# Загрузка и объединение файлов
# ────────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Загружает все CSV-файлы из директории и объединяет их в один DataFrame.
    
    Parameters
    ----------
    data_dir : str
        Путь к директории с CSV-файлами (MachineLearningCVE).
    
    Returns
    -------
    pd.DataFrame
        Объединённый датафрейм со всеми записями.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"CSV-файлы не найдены в {data_dir}")
    
    print(f"Найдено файлов: {len(csv_files)}")
    
    dfs = []
    for fpath in csv_files:
        fname = Path(fpath).name
        try:
            df = pd.read_csv(fpath, encoding="utf-8", low_memory=False)
            # Нормализуем имена столбцов (убираем лишние пробелы)
            df.columns = df.columns.str.strip()
            df["_source_file"] = fname
            dfs.append(df)
            print(f"  ✓ {fname}: {len(df):,} записей, "
                  f"атак: {(df['Label'] != BENIGN_LABEL.strip()).sum():,}")
        except Exception as e:
            print(f"  ✗ {fname}: ошибка — {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nИтого записей: {len(combined):,}")
    return combined


# ────────────────────────────────────────────────────────────────────────────────
# Очистка данных
# ────────────────────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовая очистка: удаление дубликатов, инфинитов, NaN.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    """
    n_before = len(df)
    
    # Заменяем inf на NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Удаляем строки с NaN в ключевых числовых столбцах
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.dropna(subset=numeric_cols, how="all")
    
    # Удаляем дубликаты
    df = df.drop_duplicates()
    
    n_after = len(df)
    print(f"Очистка: {n_before:,} → {n_after:,} записей "
          f"(удалено {n_before - n_after:,})")
    return df.reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────────
# Создание суррогатной временной оси
# ────────────────────────────────────────────────────────────────────────────────

def assign_surrogate_time(df: pd.DataFrame,
                           delta_t_sec: float = DELTA_T_SECONDS) -> pd.DataFrame:
    """
    Поскольку в датасете нет явного timestamp, создаём суррогатное время:
    - Каждый файл соответствует отдельному «дню» (дни упорядочены по имени файла).
    - Внутри файла строки упорядочены в порядке чтения.
    - Каждой строке присваивается дробное время (в секундах) внутри дня.
    
    Это позволяет получить монотонную временную ось для агрегирования.
    
    Parameters
    ----------
    df : pd.DataFrame
    delta_t_sec : float
        Размер окна в секундах (для справки).
    
    Returns
    -------
    pd.DataFrame
        С добавленным столбцом 't_sec' — время в секундах.
    """
    # Порядок файлов по дням недели
    day_order = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
    }
    
    def get_day_offset(source_file: str) -> int:
        for day, offset in day_order.items():
            if day in source_file:
                return offset
        return 5  # неизвестный файл — в конец
    
    df = df.copy()
    df["_day_offset"] = df["_source_file"].apply(get_day_offset)
    
    # Сортируем внутри каждого файла по Flow Duration (суррогат временного порядка)
    # Это даёт более реалистичный временной ряд, чем порядок строк в CSV
    if "Flow Duration" in df.columns:
        df = df.sort_values(["_source_file", "Flow Duration"]).reset_index(drop=True)
    
    # Внутри каждого файла нумеруем строки
    df["_row_in_file"] = df.groupby("_source_file").cumcount()
    df["_file_size"] = df.groupby("_source_file")["_source_file"].transform("count")
    
    # Рабочий день = 8 часов = 28800 секунд
    WORKDAY_SEC = 8 * 3600
    
    df["t_sec"] = (
        df["_day_offset"] * WORKDAY_SEC
        + (df["_row_in_file"] / df["_file_size"]) * WORKDAY_SEC
    )
    
    return df


# ────────────────────────────────────────────────────────────────────────────────
# Агрегирование в временные окна → N_k
# ────────────────────────────────────────────────────────────────────────────────

def aggregate_to_windows(df: pd.DataFrame,
                          delta_t_sec: float = DELTA_T_SECONDS
                          ) -> pd.DataFrame:
    """
    Агрегирует поток записей в временные окна размером Δt.
    
    Для каждого окна k вычисляем:
      - N_k   : число атаковых flow-записей (наблюдаемое D(t))
      - N_total_k : общее число записей
      - attack_rate_k : доля атак (N_k / N_total_k)
    
    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с колонкой 't_sec' и 'Label'.
    delta_t_sec : float
        Размер временного окна в секундах.
    
    Returns
    -------
    pd.DataFrame
        Агрегированный временной ряд с колонками:
        ['t_bin', 't_center', 'N_k', 'N_total_k', 'attack_rate', 'attack_types']
    """
    df = df.copy()
    df["Label"] = df["Label"].astype(str).str.strip()
    df["is_attack"] = (df["Label"] != BENIGN_LABEL).astype(int)
    
    # Бинируем по времени
    t_max = df["t_sec"].max()
    bins = np.arange(0, t_max + delta_t_sec, delta_t_sec)
    df["t_bin"] = pd.cut(df["t_sec"], bins=bins, labels=False, right=False)
    
    # Агрегируем
    agg = df.groupby("t_bin").agg(
        N_k=("is_attack", "sum"),
        N_total_k=("is_attack", "count"),
        attack_types=("Label", lambda x: list(x[x != BENIGN_LABEL].unique()))
    ).reset_index()
    
    # Центр окна в секундах
    agg["t_center"] = (agg["t_bin"] + 0.5) * delta_t_sec
    agg["attack_rate"] = agg["N_k"] / agg["N_total_k"].clip(lower=1)
    
    # Добавляем пустые окна (где не было записей)
    all_bins = pd.DataFrame({"t_bin": np.arange(len(bins) - 1)})
    agg = all_bins.merge(agg, on="t_bin", how="left").fillna(0)
    agg["t_center"] = (agg["t_bin"] + 0.5) * delta_t_sec
    agg["attack_types"] = agg["attack_types"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    print(f"Временных окон: {len(agg)}, Δt = {delta_t_sec}с")
    print(f"Среднее N_k: {agg['N_k'].mean():.1f}, "
          f"Макс N_k: {agg['N_k'].max():.0f}")
    
    return agg.sort_values("t_bin").reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────────
# Нормализация для подачи в нейросеть
# ────────────────────────────────────────────────────────────────────────────────

def normalize_series(series: pd.DataFrame) -> tuple:
    """
    Нормализует временной ряд N_k для обучения PINN.
    
    Стратегия:
    - t нормализуем в [0, 1]
    - D(t) = N_k нормализуем в [0, 1] через max-scaling
    
    Returns
    -------
    (t_norm, D_norm, t_scale, D_scale) : нормализованные данные + масштабы
    """
    t = series["t_center"].values.astype(np.float32)
    D = series["N_k"].values.astype(np.float32)
    
    t_scale = t.max() if t.max() > 0 else 1.0
    D_scale = D.max() if D.max() > 0 else 1.0
    
    t_norm = t / t_scale
    D_norm = D / D_scale
    
    return t_norm, D_norm, float(t_scale), float(D_scale)


# ────────────────────────────────────────────────────────────────────────────────
# Главная функция pipeline
# ────────────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(data_dir: str,
                         delta_t_sec: float = DELTA_T_SECONDS,
                         save_path: str = None) -> dict:
    """
    Полный pipeline предобработки данных.
    
    Parameters
    ----------
    data_dir : str
        Директория с CSV-файлами.
    delta_t_sec : float
        Размер временного окна.
    save_path : str, optional
        Путь для сохранения результата (NPZ-файл).
    
    Returns
    -------
    dict с ключами:
        't_norm'   : нормализованное время [0,1]
        'D_norm'   : нормализованная интенсивность атак [0,1]
        't_scale'  : масштаб времени (секунды)
        'D_scale'  : масштаб D(t) (макс. число атак в окне)
        'agg_df'   : агрегированный DataFrame
    """
    print("=" * 60)
    print("ПРЕДОБРАБОТКА ДАННЫХ CIC-IDS2017")
    print("=" * 60)
    
    # 1. Загрузка
    df = load_dataset(data_dir)
    
    # 2. Очистка
    df = clean_dataframe(df)
    
    # 3. Суррогатное время
    df = assign_surrogate_time(df, delta_t_sec)
    
    # 4. Агрегирование
    agg_df = aggregate_to_windows(df, delta_t_sec)
    
    # 5. Нормализация
    t_norm, D_norm, t_scale, D_scale = normalize_series(agg_df)
    
    print(f"\nt ∈ [0, {t_scale:.0f}с], нормализовано в [0, 1]")
    print(f"D ∈ [0, {D_scale:.0f}], нормализовано в [0, 1]")
    
    result = {
        "t_norm": t_norm,
        "D_norm": D_norm,
        "t_scale": t_scale,
        "D_scale": D_scale,
        "agg_df": agg_df,
        "delta_t_sec": delta_t_sec,
    }
    
    # Сохранение
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez(save_path,
                 t_norm=t_norm,
                 D_norm=D_norm,
                 t_scale=np.array([t_scale]),
                 D_scale=np.array([D_scale]),
                 delta_t_sec=np.array([delta_t_sec]))
        print(f"\nДанные сохранены: {save_path}")

        if save_preprocessing_artifacts is not None:
            output_dir = os.path.dirname(os.path.abspath(save_path))
            save_preprocessing_artifacts(
                agg_df=agg_df,
                t_norm=t_norm,
                D_norm=D_norm,
                t_scale=t_scale,
                D_scale=D_scale,
                delta_t_sec=delta_t_sec,
                output_dir=output_dir,
            )
            print(f"Артефакты предобработки сохранены в: {output_dir}")
    
    return result


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    result = preprocess_pipeline(data_dir, save_path="preprocessed_data.npz")
    print("\nГотово.")
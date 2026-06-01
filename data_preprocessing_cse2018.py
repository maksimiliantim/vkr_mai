"""
data_preprocessing_cse2018.py
=============================
Предобработка CSE-CIC-IDS2018 для дополнительного эксперимента ВКР.

Ключевая версия: используется реальный Timestamp внутри каждого CSV, но календарные
разрывы между файлами удаляются. Каждый CSV рассматривается как отдельный сценарный
день, файлы упорядочиваются по минимальному Timestamp, а затем склеиваются в
последовательную рабочую временную ось.

Это устраняет проблему растяжения ряда на несколько недель/лет и делает временной
ряд пригодным для восстановления и первичной прогнозной проверки.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_LABEL_COL = "Label"
DEFAULT_TIME_COL = "Timestamp"
BENIGN_VALUES = {"benign", "normal"}
DEFAULT_N_WINDOWS = 2400
MIN_VALID_DATE = pd.Timestamp("2018-02-01")
MAX_VALID_DATE = pd.Timestamp("2018-03-31 23:59:59")


def _ensure_dirs(output_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(output_dir)
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    exports_dir = root / "exports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir, exports_dir


def infer_column(columns: Iterable[str], candidates: list[str], name: str) -> str:
    cols = list(columns)
    stripped = {c.strip(): c for c in cols}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.strip() in stripped:
            return stripped[candidate.strip()]
    lowered = {c.lower().strip(): c for c in cols}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lowered:
            return lowered[key]
    raise ValueError(f"Не найден столбец {name}. Доступные колонки: {cols}")


def list_csv_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    if data_dir.is_file() and data_dir.suffix.lower() == ".csv":
        return [data_dir]
    if not data_dir.exists():
        raise FileNotFoundError(f"Путь не найден: {data_dir}")
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV-файлы не найдены: {data_dir}")
    return files


def _parse_timestamp(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    dt = pd.to_datetime(raw, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    return dt


def _valid_rows(dt: pd.Series, labels: pd.Series) -> pd.Series:
    labels_clean = labels.astype(str).str.strip()
    return (
        dt.notna()
        & (dt >= MIN_VALID_DATE)
        & (dt <= MAX_VALID_DATE)
        & (labels_clean != "Label")
        & (labels_clean != "")
        & (labels_clean.str.lower() != "nan")
    )


def scan_files(
    files: list[Path],
    chunksize: int = 500_000,
) -> tuple[list[dict], pd.DataFrame]:
    metadata: list[dict] = []
    label_rows: list[dict] = []

    for path in files:
        print(f"Сканирую: {path.name}")
        header = pd.read_csv(path, nrows=0)
        time_col = infer_column(header.columns, [DEFAULT_TIME_COL, "timestamp", "Time", "time"], "времени")
        label_col = infer_column(header.columns, [DEFAULT_LABEL_COL, "label", "Category", "category"], "метки")

        file_min = None
        file_max = None
        valid_rows_count = 0
        skipped_rows_count = 0
        label_counts: dict[str, int] = {}

        for chunk in pd.read_csv(path, usecols=[time_col, label_col], chunksize=chunksize, low_memory=False):
            chunk = chunk.rename(columns={time_col: "Timestamp", label_col: "Label"})
            dt = _parse_timestamp(chunk["Timestamp"])
            labels = chunk["Label"].astype(str).str.strip()
            valid = _valid_rows(dt, labels)

            skipped_rows_count += int((~valid).sum())
            if not valid.any():
                continue

            dt_valid = dt[valid]
            labels_valid = labels[valid]
            valid_rows_count += len(dt_valid)

            cmin = dt_valid.min()
            cmax = dt_valid.max()
            file_min = cmin if file_min is None else min(file_min, cmin)
            file_max = cmax if file_max is None else max(file_max, cmax)

            counts = labels_valid.value_counts()
            for label, count in counts.items():
                label_counts[str(label)] = label_counts.get(str(label), 0) + int(count)

        if file_min is None or file_max is None or valid_rows_count == 0:
            print(f"  пропуск: нет корректных строк")
            continue

        duration_sec = max(float((file_max - file_min).total_seconds()), 1.0)
        item = {
            "path": path,
            "file": path.name,
            "rows": valid_rows_count,
            "skipped_rows": skipped_rows_count,
            "t_min": file_min,
            "t_max": file_max,
            "duration_sec": duration_sec,
            "labels": label_counts,
        }
        metadata.append(item)

    metadata.sort(key=lambda x: x["t_min"])

    offset = 0.0
    for item in metadata:
        item["offset_sec"] = offset
        offset += item["duration_sec"]

        for label, count in item["labels"].items():
            label_rows.append({
                "file": item["file"],
                "rows_in_file": item["rows"],
                "skipped_rows": item["skipped_rows"],
                "t_min": item["t_min"],
                "t_max": item["t_max"],
                "duration_sec": item["duration_sec"],
                "offset_sec": item["offset_sec"],
                "label": label,
                "count": count,
            })

    if not metadata:
        raise ValueError("Не найдено корректных файлов CSE-CIC-IDS2018 после фильтрации дат")

    return metadata, pd.DataFrame(label_rows)


def aggregate_cse2018_to_windows(
    metadata: list[dict],
    n_windows: int = DEFAULT_N_WINDOWS,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    if n_windows <= 1:
        raise ValueError("n_windows должен быть больше 1")

    total_duration = max(metadata[-1]["offset_sec"] + metadata[-1]["duration_sec"], 1.0)
    N_k = np.zeros(n_windows, dtype=np.float64)
    N_total_k = np.zeros(n_windows, dtype=np.float64)
    attack_type_sets = [set() for _ in range(n_windows)]

    for item in metadata:
        path = item["path"]
        print(f"Агрегирую: {path.name}")
        header = pd.read_csv(path, nrows=0)
        time_col = infer_column(header.columns, [DEFAULT_TIME_COL, "timestamp", "Time", "time"], "времени")
        label_col = infer_column(header.columns, [DEFAULT_LABEL_COL, "label", "Category", "category"], "метки")

        file_start = item["t_min"]
        offset_sec = item["offset_sec"]

        for chunk in pd.read_csv(path, usecols=[time_col, label_col], chunksize=chunksize, low_memory=False):
            chunk = chunk.rename(columns={time_col: "Timestamp", label_col: "Label"})
            dt = _parse_timestamp(chunk["Timestamp"])
            labels = chunk["Label"].astype(str).str.strip()
            valid = _valid_rows(dt, labels)
            if not valid.any():
                continue

            dt_valid = dt[valid]
            labels_valid = labels[valid]
            local_sec = (dt_valid - file_start).dt.total_seconds().to_numpy(dtype=np.float64)
            seq_sec = offset_sec + np.clip(local_sec, 0.0, None)
            bins = np.floor(seq_sec / total_duration * n_windows).astype(np.int64)
            bins = np.clip(bins, 0, n_windows - 1)

            label_lower = labels_valid.str.lower()
            is_attack_bool = ~label_lower.isin(BENIGN_VALUES)
            is_attack = is_attack_bool.to_numpy(dtype=np.float64)

            N_total_k += np.bincount(bins, minlength=n_windows)
            N_k += np.bincount(bins, weights=is_attack, minlength=n_windows)

            attack_bins = bins[is_attack_bool.to_numpy()]
            attack_labels = labels_valid.to_numpy()[is_attack_bool.to_numpy()]
            for b, lab in zip(attack_bins, attack_labels):
                if len(attack_type_sets[int(b)]) < 20:
                    attack_type_sets[int(b)].add(str(lab))

    t_center_sec = (np.arange(n_windows, dtype=np.float64) + 0.5) / n_windows * total_duration
    agg = pd.DataFrame({
        "t_bin": np.arange(n_windows, dtype=int),
        "t_center_sec": t_center_sec,
        "N_k": N_k,
        "N_total_k": N_total_k,
    })
    agg["attack_rate"] = agg["N_k"] / np.clip(agg["N_total_k"], 1, None)
    agg["attack_types"] = [sorted(s) for s in attack_type_sets]
    return agg


def normalize_cse_series(series: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    t = series["t_center_sec"].to_numpy(dtype=np.float32)
    D = series["N_k"].to_numpy(dtype=np.float32)
    t_scale = float(t.max()) if float(t.max()) > 0 else 1.0
    D_scale = float(D.max()) if float(D.max()) > 0 else 1.0
    return t / t_scale, D / D_scale, t_scale, D_scale


def _plot_cse_series(series: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None:
        print("matplotlib не установлен: PNG-графики предобработки CSE-CIC-IDS2018 пропущены")
        return

    t_hours = series["t_center_sec"].to_numpy() / 3600

    plt.figure(figsize=(11, 4))
    plt.plot(t_hours, series["N_k"], linewidth=1.1)
    plt.xlabel("Последовательное время наблюдения, часы")
    plt.ylabel("Число атаковых записей в окне")
    plt.title("CSE-CIC-IDS2018: агрегированный временной ряд атаковой активности")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_cse_1_observed_series.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(series["N_k"], bins=50)
    plt.xlabel("N_k")
    plt.ylabel("Частота")
    plt.title("CSE-CIC-IDS2018: распределение числа атаковых записей по окнам")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_cse_2_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_cse2018_pipeline(
    data_dir: str | Path,
    output_dir: str | Path = "./results_cse2018",
    n_windows: int = DEFAULT_N_WINDOWS,
    chunksize: int = 500_000,
    save_npz: bool = True,
    file_limit: int | None = None,
    exclude_large_mb: float | None = None,
) -> dict:
    print("=" * 60)
    print("ПРЕДОБРАБОТКА CSE-CIC-IDS2018")
    print("=" * 60)

    figures_dir, tables_dir, exports_dir = _ensure_dirs(output_dir)
    files = list_csv_files(data_dir)

    if exclude_large_mb is not None:
        before = len(files)
        files = [f for f in files if f.stat().st_size / 1024**2 <= exclude_large_mb]
        print(f"Исключены файлы больше {exclude_large_mb:.0f} MB: {before - len(files)}")

    if file_limit is not None:
        files = files[:file_limit]

    print(f"CSV-файлов для обработки: {len(files)}")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")

    metadata, label_summary = scan_files(files, chunksize=chunksize)
    file_summary = pd.DataFrame([
        {
            "file": item["file"],
            "rows": item["rows"],
            "skipped_rows": item["skipped_rows"],
            "t_min": item["t_min"],
            "t_max": item["t_max"],
            "duration_sec": item["duration_sec"],
            "offset_sec": item["offset_sec"],
        }
        for item in metadata
    ])
    file_summary.to_excel(tables_dir / "cse2018_file_time_summary.xlsx", index=False)
    label_summary.to_excel(tables_dir / "cse2018_label_distribution_by_file.xlsx", index=False)

    label_counts = label_summary.groupby("label", as_index=False)["count"].sum()
    label_counts["share"] = label_counts["count"] / label_counts["count"].sum()
    label_counts = label_counts.sort_values("count", ascending=False)
    label_counts.to_excel(tables_dir / "cse2018_label_distribution.xlsx", index=False)

    total_duration = metadata[-1]["offset_sec"] + metadata[-1]["duration_sec"]
    print("\nПоследовательная временная ось по файлам:")
    print(file_summary[["file", "t_min", "t_max", "duration_sec", "offset_sec"]].to_string(index=False))
    print("\nРаспределение меток, первые 15:")
    print(label_counts.head(15).to_string(index=False))
    print(f"\nИтоговая длительность после удаления календарных разрывов: {total_duration:.0f} сек. ({total_duration / 3600:.2f} ч)")

    series = aggregate_cse2018_to_windows(metadata, n_windows=n_windows, chunksize=chunksize)
    t_norm, D_norm, t_scale, D_scale = normalize_cse_series(series)

    print(f"\nОкон: {len(series)}, N_k max = {D_scale:.0f}, N_k mean = {series['N_k'].mean():.2f}")
    print(f"t_scale = {t_scale:.0f} сек., D_scale = {D_scale:.0f}")

    series.to_csv(exports_dir / "cse2018_aggregated_series.csv", index=False, encoding="utf-8-sig")
    series.head(200).to_excel(tables_dir / "cse2018_aggregated_series_preview.xlsx", index=False)
    _plot_cse_series(series, figures_dir)

    if save_npz:
        np.savez(
            Path(output_dir) / "cse2018_preprocessed.npz",
            t_norm=t_norm,
            D_norm=D_norm,
            t_scale=np.array([t_scale], dtype=np.float32),
            D_scale=np.array([D_scale], dtype=np.float32),
            n_windows=np.array([n_windows], dtype=np.int32),
        )

    return {
        "t_norm": t_norm,
        "D_norm": D_norm,
        "t_scale": t_scale,
        "D_scale": D_scale,
        "agg_df": series,
        "label_counts": label_counts,
        "label_summary": label_summary,
        "file_summary": file_summary,
        "files": [item["path"] for item in metadata],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Предобработка CSE-CIC-IDS2018 для PINN")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./results_cse2018")
    parser.add_argument("--n_windows", type=int, default=DEFAULT_N_WINDOWS)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--file_limit", type=int, default=None)
    parser.add_argument("--exclude_large_mb", type=float, default=None)
    args = parser.parse_args()

    preprocess_cse2018_pipeline(
        args.data_dir,
        args.output_dir,
        args.n_windows,
        chunksize=args.chunksize,
        file_limit=args.file_limit,
        exclude_large_mb=args.exclude_large_mb,
    )

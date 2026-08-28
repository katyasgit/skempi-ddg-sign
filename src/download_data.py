"""Скачивание базы SKEMPI 2.0.

SKEMPI 2.0 — база экспериментальных измерений аффинности связывания
белок-белковых комплексов: дикий тип против мутантов, ~7 тысяч записей.
Источник: https://life.bsc.es/pid/skempi2
"""
from pathlib import Path
import urllib.request

URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "skempi_v2.csv"


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        print(f"Файл уже скачан: {RAW_PATH}")
        return
    print(f"Скачиваю {URL} ...")
    urllib.request.urlretrieve(URL, RAW_PATH)
    size_mb = RAW_PATH.stat().st_size / 1e6
    print(f"Готово: {RAW_PATH} ({size_mb:.1f} МБ)")


if __name__ == "__main__":
    main()

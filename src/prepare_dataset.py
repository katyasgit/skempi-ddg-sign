"""Подготовка датасета из SKEMPI 2.0: расчёт ddG, метка класса и признаки.

Вход:  data/raw/skempi_v2.csv   (кладёт туда скрипт download_data.py)
Выход: data/processed/dataset.csv

Общий ход рассуждений описан в README.md, здесь — комментарии по шагам.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "skempi_v2.csv"
OUT_PATH = ROOT / "data" / "processed" / "dataset.csv"

# Газовая постоянная в ккал/(моль·K), чтобы ddG получался в привычных ккал/моль
R = 1.9872e-3
DEFAULT_T = 298.0  # если температура в записи не указана, берём комнатную

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Гидрофобность аминокислот по шкале Кайта — Дулиттла
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Объём остатка в кубических ангстремах (Zamyatnin, 1972)
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Формальный заряд боковой цепи при pH~7; гистидин заряжен лишь частично
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

# Положение мутируемого остатка относительно интерфейса (аннотация SKEMPI):
# COR/SUP/RIM — ядро/опора/край интерфейса, INT — внутри белка, SUR — поверхность
LOCATIONS = ["COR", "SUP", "RIM", "INT", "SUR"]


def parse_temperature(value) -> float:
    """Достаёт число из строки вида '298' или '298(assumed)'; иначе 298."""
    m = re.search(r"\d+(\.\d+)?", str(value))
    t = float(m.group()) if m else DEFAULT_T
    # страховка от опечаток и значений не в кельвинах
    return t if 250.0 <= t <= 400.0 else DEFAULT_T


def compute_ddg(df: pd.DataFrame) -> pd.Series:
    """ddG = R·T·ln(Kd_mut / Kd_wt), в ккал/моль.

    Свободная энергия связывания: dG = R·T·ln(Kd); ddG = dG_mut − dG_wt.
    Положительный ddG означает, что мутация ослабляет связывание
    (дестабилизирует комплекс), отрицательный — усиливает.
    """
    t = df["Temperature"].map(parse_temperature)
    kd_mut = pd.to_numeric(df["Affinity_mut_parsed"], errors="coerce")
    kd_wt = pd.to_numeric(df["Affinity_wt_parsed"], errors="coerce")
    return R * t * (np.log(kd_mut) - np.log(kd_wt))


def mutation_features(mutations: str, locations: str) -> dict | None:
    """Признаки одной записи по списку мутаций вида 'LI38G' или 'LI38G,KI27A'.

    Формат мутации в SKEMPI: 'LI38G' = Leu (дикий тип), цепь I, позиция 38,
    Gly (мутант). Для признаков нужны только первая и последняя буквы.
    Возвращает None, если запись не удаётся разобрать.
    """
    feats: dict[str, float] = {f"wt_{aa}": 0.0 for aa in AMINO_ACIDS}
    feats |= {f"mut_{aa}": 0.0 for aa in AMINO_ACIDS}
    feats |= {f"loc_{loc}": 0.0 for loc in LOCATIONS}
    feats |= {"d_hydropathy": 0.0, "d_volume": 0.0, "d_charge": 0.0}

    muts = [m.strip() for m in str(mutations).split(",") if m.strip()]
    if not muts:
        return None
    for m in muts:
        wt, mut = m[0], m[-1]
        if wt not in AMINO_ACIDS or mut not in AMINO_ACIDS:
            return None
        # какие аминокислоты заменяются и на что (счётчики, т.к. мутаций может быть несколько)
        feats[f"wt_{wt}"] += 1
        feats[f"mut_{mut}"] += 1
        # физико-химические сдвиги «мутант минус дикий тип», суммируем по мутациям
        feats["d_hydropathy"] += HYDROPATHY[mut] - HYDROPATHY[wt]
        feats["d_volume"] += VOLUME[mut] - VOLUME[wt]
        feats["d_charge"] += CHARGE.get(mut, 0.0) - CHARGE.get(wt, 0.0)
    for loc in str(locations).split(","):
        loc = loc.strip().upper()
        if loc in LOCATIONS:
            feats[f"loc_{loc}"] += 1
    feats["n_mutations"] = float(len(muts))
    return feats


def main() -> None:
    # В SKEMPI разделитель — точка с запятой
    df = pd.read_csv(RAW_PATH, sep=";", low_memory=False)
    print(f"Записей в SKEMPI 2.0: {len(df)}")

    df["ddg"] = compute_ddg(df)
    df["complex"] = df["#Pdb"].str[:4]  # первые 4 символа — PDB-код комплекса

    # 1) без измеренной аффинности мутанта или дикого типа ddG не посчитать
    df = df.dropna(subset=["ddg"])
    print(f"С измеренной аффинностью (есть ddG): {len(df)}")

    # 2) одна и та же мутация нередко измерена в нескольких работах —
    #    сворачиваем дубликаты медианой, чтобы одинаковые примеры
    #    не «протекали» одновременно в train и в test
    grouped = df.groupby(["#Pdb", "Mutation(s)_cleaned"], as_index=False).agg(
        ddg=("ddg", "median"),
        locations=("iMutation_Location(s)", "first"),
        complex=("complex", "first"),
    )
    print(f"Уникальных пар (комплекс, мутация): {len(grouped)}")

    # 3) при ddG = 0 знак не определён — такие записи убираем
    grouped = grouped[grouped["ddg"] != 0].reset_index(drop=True)

    # 4) целевая переменная: 1 — дестабилизирующая мутация (ddG > 0)
    grouped["label"] = (grouped["ddg"] > 0).astype(int)

    # 5) признаки из строки мутаций и аннотации положения
    feats = [
        mutation_features(m, l)
        for m, l in zip(grouped["Mutation(s)_cleaned"], grouped["locations"])
    ]
    ok = [f is not None for f in feats]
    n_bad = len(feats) - sum(ok)
    if n_bad:
        print(f"Не удалось разобрать мутации в {n_bad} записях — пропускаю их")
    feat_df = pd.DataFrame([f for f in feats if f is not None])

    out = pd.concat(
        [
            grouped.loc[ok, ["#Pdb", "Mutation(s)_cleaned", "complex", "ddg", "label"]]
            .rename(columns={"#Pdb": "pdb_entry", "Mutation(s)_cleaned": "mutations"})
            .reset_index(drop=True),
            feat_df,
        ],
        axis=1,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Итоговый датасет: {len(out)} строк, {feat_df.shape[1]} признаков")
    print(f"Уникальных комплексов: {out['complex'].nunique()}")
    print(f"Доля дестабилизирующих мутаций (label=1): {out['label'].mean():.1%}")
    print(f"Сохранено в {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Обучение и оценка моделей, предсказывающих знак ddG.

Вход:  data/processed/dataset.csv  (готовит скрипт prepare_dataset.py)
Выход: results/metrics.md, results/metrics.json, results/figures/*.png

Ключевое решение — валидация с группировкой по PDB-комплексу (GroupKFold):
у одного комплекса в базе десятки-сотни мутаций, и при обычном случайном
разбиении один и тот же комплекс попадает и в train, и в test. Тогда модель
может просто «запомнить» комплекс, и оценка окажется завышенной. Поэтому все
мутации одного комплекса всегда целиком уходят либо в train, либо в test.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, дисплей не нужен

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_validate,
    learning_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dataset.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

META_COLS = ["pdb_entry", "mutations", "complex", "ddg", "label"]
SCORING = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
RANDOM_STATE = 42

# ----- оформление графиков: спокойная палитра, подписи по-русски -----
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
BLUES_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": BASELINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=META_COLS)
    y = df["label"].to_numpy()
    groups = df["complex"].to_numpy()  # группы для честного разбиения
    return df, X, y, groups


def make_models() -> dict:
    """Три модели по нарастанию сложности.

    Baseline всегда отвечает самым частым классом — это планка «модель
    ничему не научилась». Логистическая регрессия — простейшая линейная
    модель (признаки предварительно стандартизуем). Случайный лес умеет
    ловить нелинейности и взаимодействия признаков.
    """
    return {
        "Baseline (частый класс)": DummyClassifier(strategy="most_frequent"),
        "Логистическая регрессия": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Случайный лес": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_models(models, X, y, groups) -> dict:
    """5-кратная кросс-валидация с группировкой по комплексам."""
    cv = GroupKFold(n_splits=5)
    results: dict[str, dict] = {}
    for name, model in models.items():
        cvr = cross_validate(model, X, y, groups=groups, cv=cv,
                             scoring=SCORING, n_jobs=-1)
        results[name] = {
            m: {"mean": float(cvr[f"test_{m}"].mean()),
                "std": float(cvr[f"test_{m}"].std())}
            for m in SCORING
        }
    return results


# ----------------------------- графики -----------------------------

def plot_ddg_distribution(df: pd.DataFrame) -> None:
    """Распределение ddG: слева от нуля — стабилизирующие мутации, справа —
    дестабилизирующие. Заодно видно баланс классов."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-6, 10, 81)
    ddg = df["ddg"].clip(-6, 10)  # редкие хвосты поджимаем к границам рисунка
    ax.hist(ddg[df["label"] == 0], bins=bins, color=BLUE, alpha=0.9,
            label=f"стабилизирующие, ddG < 0 ({(df['label'] == 0).mean():.0%})")
    ax.hist(ddg[df["label"] == 1], bins=bins, color=RED, alpha=0.9,
            label=f"дестабилизирующие, ddG > 0 ({(df['label'] == 1).mean():.0%})")
    ax.axvline(0, color=BASELINE, linewidth=1.2)
    ax.set_xlabel("ddG, ккал/моль")
    ax.set_ylabel("Число мутаций")
    ax.set_title("Распределение ddG в подготовленном датасете")
    ax.legend(loc="upper right")
    fig.savefig(FIGURES / "ddg_distribution.png")
    plt.close(fig)


def plot_learning_curve(model, X, y, groups) -> None:
    """Главная иллюстрация того, что модель обучается: качество на
    отложенных комплексах растёт с объёмом обучающей выборки."""
    cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, groups=groups, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="balanced_accuracy", shuffle=True,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for scores, color, label in [
        (train_scores, BLUE, "обучающая выборка"),
        (val_scores, ORANGE, "отложенные комплексы"),
    ]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, marker="o", markersize=5, color=color, label=label)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15,
                        linewidth=0)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(sizes[-1], 0.505, "случайное угадывание", ha="right", va="bottom",
            fontsize=9, color=MUTED)
    ax.set_xlabel("Размер обучающей выборки, записей")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Кривая обучения (случайный лес)")
    ax.legend(loc="center right")
    fig.savefig(FIGURES / "learning_curve.png")
    plt.close(fig)


def plot_roc_curves(fitted: dict, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    colors = {"Логистическая регрессия": BLUE, "Случайный лес": ORANGE}
    for name, model in fitted.items():
        if name not in colors:
            continue  # у baseline осмысленной ROC-кривой нет
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr = roc_curve(y_test, proba)[:2]
        auc = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, color=colors[name], label=f"{name} (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2,
            label="случайное угадывание")
    ax.set_xlabel("Доля ложных срабатываний (FPR)")
    ax.set_ylabel("Доля найденных положительных (TPR)")
    ax.set_title("ROC-кривые на отложенных комплексах")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(FIGURES / "roc_curves.png")
    plt.close(fig)


def plot_confusion_matrix(model, X_test, y_test) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    share = cm / cm.sum(axis=1, keepdims=True)  # доли по строкам
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.grid(False)
    ax.imshow(share, cmap=BLUES_CMAP, vmin=0, vmax=1)
    ticks = ["стабилизирующая\n(ddG < 0)", "дестабилизирующая\n(ddG > 0)"]
    ax.set_xticks([0, 1], ticks)
    ax.set_yticks([0, 1], ticks)
    for i in range(2):
        for j in range(2):
            color = "#ffffff" if share[i, j] > 0.55 else INK
            ax.text(j, i, f"{cm[i, j]}\n({share[i, j]:.0%})",
                    ha="center", va="center", color=color, fontsize=12)
    ax.set_xlabel("Предсказание модели")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок (случайный лес)")
    fig.savefig(FIGURES / "confusion_matrix.png")
    plt.close(fig)


def pretty_feature_name(name: str) -> str:
    """Человекочитаемые русские подписи признаков для графика важностей."""
    fixed = {
        "d_hydropathy": "Δ гидрофобности",
        "d_volume": "Δ объёма остатка",
        "d_charge": "Δ заряда",
        "n_mutations": "число мутаций",
        "loc_COR": "позиция: ядро интерфейса (COR)",
        "loc_SUP": "позиция: опора интерфейса (SUP)",
        "loc_RIM": "позиция: край интерфейса (RIM)",
        "loc_INT": "позиция: внутри белка (INT)",
        "loc_SUR": "позиция: поверхность (SUR)",
    }
    if name in fixed:
        return fixed[name]
    if name.startswith("wt_"):
        return f"дикий тип: {name[3:]}"
    if name.startswith("mut_"):
        return f"мутант: {name[4:]}"
    return name


def plot_feature_importances(model, feature_names) -> None:
    imp = pd.Series(model.feature_importances_, index=feature_names)
    top = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([pretty_feature_name(n) for n in top.index], top.values,
            color=BLUE, height=0.6)
    ax.set_xlabel("Важность признака (случайный лес)")
    ax.set_title("Топ-15 признаков по важности")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "feature_importances.png")
    plt.close(fig)


# ----------------------------- отчёт -----------------------------

def write_metrics(results: dict, n_rows: int, n_complexes: int) -> None:
    metric_titles = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }
    lines = [
        "# Метрики",
        "",
        f"5-кратная кросс-валидация **GroupKFold по PDB-комплексам** "
        f"({n_rows} записей, {n_complexes} комплексов): все мутации одного "
        "комплекса попадают либо в train, либо в test, поэтому оценка не "
        "завышена из-за «знакомых» комплексов. В таблице — среднее ± "
        "стандартное отклонение по фолдам.",
        "",
        "| Модель | " + " | ".join(metric_titles.values()) + " |",
        "|---|" + "---|" * len(metric_titles),
    ]
    for name, metrics in results.items():
        cells = [
            f"{metrics[m]['mean']:.3f} ± {metrics[m]['std']:.3f}"
            for m in metric_titles
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    df, X, y, groups = load_data()
    print(f"Данных: {len(df)} записей, {X.shape[1]} признаков, "
          f"{df['complex'].nunique()} комплексов")

    plot_ddg_distribution(df)

    # --- кросс-валидация трёх моделей ---
    models = make_models()
    results = evaluate_models(models, X, y, groups)
    write_metrics(results, len(df), df["complex"].nunique())
    for name, metrics in results.items():
        row = ", ".join(f"{m}={metrics[m]['mean']:.3f}±{metrics[m]['std']:.3f}"
                        for m in SCORING)
        print(f"{name}: {row}")

    # --- кривая обучения: качество растёт с объёмом данных ---
    plot_learning_curve(make_models()["Случайный лес"], X, y, groups)

    # --- один отложенный сплит для наглядных картинок (ROC, ошибки, важности) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    fitted = {}
    for name, model in make_models().items():
        fitted[name] = model.fit(X_tr, y_tr)
    plot_roc_curves(fitted, X_te, y_te)
    plot_confusion_matrix(fitted["Случайный лес"], X_te, y_te)
    plot_feature_importances(fitted["Случайный лес"], X.columns)

    # --- короткий вывод в консоль ---
    ba_model = results["Случайный лес"]["balanced_accuracy"]["mean"]
    ba_base = results["Baseline (частый класс)"]["balanced_accuracy"]["mean"]
    auc = results["Случайный лес"]["roc_auc"]["mean"]
    print(
        f"\nВывод: balanced accuracy случайного леса {ba_model:.3f} против "
        f"{ba_base:.3f} у baseline, ROC-AUC {auc:.3f} > 0.5 — модель обучается."
    )
    print(f"Метрики и графики сохранены в {RESULTS}/")


if __name__ == "__main__":
    main()

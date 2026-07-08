# check_label_distribution.py
import pandas as pd
from pathlib import Path

# NAS のベースパスを実環境に合わせて変更してください
NAS_PATH = "/nas/data_2/ueno1212/light"

paths = {
    "real_train":  f"{NAS_PATH}/data/keita_2026_0617/1_tsunoda_1_train/bbox_label.csv",
    "real_val":    f"{NAS_PATH}/data/keita_2026_0617/1_tsunoda_1_val/bbox_label.csv",
    "real_test":   f"{NAS_PATH}/data/keita_2026_0617/1_tsunoda_1_test/bbox_label.csv",
    "sim_loop1":   f"{NAS_PATH}/data/sim_lecture_02/loop1/bbox_label.csv",
    "sim_loop2":   f"{NAS_PATH}/data/sim_lecture_03/loop1/bbox_label.csv",
}

print("=" * 70)
print(f"{'Dataset':<15} | {'Total':>8} | Class distribution")
print("=" * 70)

all_stats = {}
for name, path in paths.items():
    if not Path(path).exists():
        print(f"{name:<15} | (file not found: {path})")
        continue
    df = pd.read_csv(path)
    total = len(df)
    label_counts = df['label'].value_counts().sort_index().to_dict()
    all_stats[name] = label_counts
    counts_str = ", ".join([f"C{k}={v}" for k, v in label_counts.items()])
    print(f"{name:<15} | {total:>8} | {counts_str}")

print("=" * 70)

# クラス別の割合も表示
print("\n【クラス別割合 (%)】")
print(f"{'Dataset':<15} | {'C0':>7} | {'C1':>7} | {'C2':>7} | {'C3':>7}")
print("-" * 55)
for name, counts in all_stats.items():
    total = sum(counts.values())
    ratios = []
    for c in [0, 1, 2, 3]:
        r = counts.get(c, 0) / total * 100 if total > 0 else 0
        ratios.append(f"{r:>6.1f}%")
    print(f"{name:<15} | {ratios[0]} | {ratios[1]} | {ratios[2]} | {ratios[3]}")

# 学習に使われる train データ全体（実環境 + Sim2種）でのクラス分布も見る
print("\n【学習データ全体（real_train + sim_loop1 + sim_loop2）でのクラス分布】")
combined = {}
for name in ["real_train", "sim_loop1", "sim_loop2"]:
    if name in all_stats:
        for c, v in all_stats[name].items():
            combined[c] = combined.get(c, 0) + v
total = sum(combined.values())
for c in sorted(combined.keys()):
    print(f"  Class {c}: {combined[c]:>5} ({combined[c]/total*100:.1f}%)")

# 実環境 train と test でクラス構成が大きく違わないかも確認
print("\n【実環境 train と test の Class 2 割合の比較】")
for name in ["real_train", "real_val", "real_test"]:
    if name in all_stats:
        total = sum(all_stats[name].values())
        c2 = all_stats[name].get(2, 0)
        print(f"  {name}: Class 2 = {c2}/{total} ({c2/total*100:.1f}%)")

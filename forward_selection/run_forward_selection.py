#!/usr/bin/env python3
"""
軸A(光源数, 6ケース)・軸B(光源間隔, 8ケース)それぞれについて、
実データ train/val/test(固定)に対象simケースを混ぜたときの精度変化を
逐次前向き選択(greedy forward selection)で検証するドライバ。

選定ルール:
  各ステップで test_ap が最大の候補を採用。
  test_ap が同値の場合は test_ap50、それでも同値なら test_ap75 で決着。
  それでも決まらない場合のみ自動決定せず、4指標を表示して停止する
  (state json の該当ステップに "winner" を手動で書いて再実行すれば続きから進む)。

valの4指標の測り方:
  main.py は --test に渡したデータセットに対してのみ、学習後にベストモデルで
  評価し、それを --n_runs 回平均した結果 (test_accuracy/test_ap/test_ap50/test_ap75)
  を log.csv の "AVG_{n_runs}_RUNS" 行に書き出す。--val 側にはこの平均化ロジックが
  無いため、選定用の学習だけ --val と --test の両方に実valセットを渡すことで、
  main.py を一切改修せずにvalの4指標を取得する。
  --val は早期終了・チェックポイント選定に使われるだけなので、この差し替えは
  学習の挙動そのものには影響しない。

最終確認:
  各サイズ(1..len(cases))で選ばれたベスト組み合わせについて、--test を本物の
  実testセットに差し替えて1回だけ再学習し、報告用のtest指標を記録する
  (選定には一切使わない)。サイズ=全ケース数のときは候補が1通りしかないため
  選定自体を行わず、そのまま最終確認に進む。

再開性:
  進捗は state_<axis>.json に保存する。各候補の学習結果は log.csv に
  "AVG_{n_runs}_RUNS" 行が既にあればスキップする(再学習しない)ので、
  GPUが途中で落ちても再実行で続きから進められる。

使い方:
  python run_forward_selection.py --axis A          # 実行(GPU0にCUDA_VISIBLE_DEVICESを設定)
  python run_forward_selection.py --axis B          # 実行(GPU1にCUDA_VISIBLE_DEVICESを設定)
  python run_forward_selection.py --axis A --dry-run  # コマンドとexp_nameの確認のみ、学習はしない
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # sim_var/
MAIN_PY = REPO_ROOT / "Illuminance_bbox" / "main.py"
NAS_ROOT = "/nas/data_2/ueno1212/light"
NAS_DATA_ROOT = Path(NAS_ROOT) / "data"

REAL_TRAIN = "keita_2026_0617/1_tsunoda_1_train"
REAL_VAL = "keita_2026_0617/1_tsunoda_1_val"
REAL_TEST = "keita_2026_0617/1_tsunoda_1_test"

AXIS_CASES = {
    "A": [
        "sim_test_03/A/A_N1_1x1",
        "sim_test_03/A/A_N4_2x2",
        "sim_test_03/A/A_N9_3x3",
        "sim_test_03/A/A_N16_4x4",
        "sim_test_03/A/A_N25_5x5",
        "sim_test_03/A/A_N36_6x6",
    ],
    "B": [
        "sim_test_03/B/B_d0.5_N9",
        "sim_test_03/B/B_d0.75_N9",
        "sim_test_03/B/B_d1.0_N9",
        "sim_test_03/B/B_d1.25_N9",
        "sim_test_03/B/B_d1.5_N9",
        "sim_test_03/B/B_d2.0_N9",
        "sim_test_03/B/B_d2.5_N9",
        "sim_test_03/B/B_d3.0_N9",
    ],
}

# --gpu 引数は main.py の init_distributed_mode() 内で常に 0 に上書きされ
# 出力先は必ず ".../GPU0/" 固定になる(main.py側の既存挙動)。
# 物理GPUの選択は CUDA_VISIBLE_DEVICES のみで行う。
AXIS_CUDA_VISIBLE_DEVICES = {"A": "0", "B": "1"}

# device3.sh / device4.sh (environment none) と同一のテンプレート。
# sim組み合わせ(--train, --val, --test, --exp_name)以外はここで固定する。
TEMPLATE_ARGS = [
    "--batch_size", "256",
    "--epochs", "1000000",
    "--dataset_file", "illuminance",
    "--backbone", "illuminance",
    "--early_stopping_patience", "50",
    "--standardize",
    "--n_runs", "3",
    "--window_size", "41",
    "--sub_window_size", "37",
    "--stride", "1",
    "--scale", "full",
    "--model_mode", "time_sensor",
    "--num_queries", "10",
    "--enc_layers", "3",
    "--dec_layers", "3",
    "--num_classes", "4",
    "--environment", "none",
    "--num_sensors", "36",
    "--k_neighbors", "36",
    "--sensor_dropout_rate", "0.1",
    "--random_sensors",
]
N_RUNS = 3
AVG_EPOCH_LABEL = f"AVG_{N_RUNS}_RUNS"
METRICS = ["test_accuracy", "test_ap", "test_ap50", "test_ap75"]
# 選定に使う指標。上から順に同値のときのタイブレークとして使う。
DECISION_METRICS = ["test_ap", "test_ap50", "test_ap75"]


def case_short(name):
    return name.rsplit("/", 1)[-1]


def output_log_txt(exp_name):
    # main.py の args.output_dir 組み立てロジック(scale/model_mode/k_neighbors/
    # num_sensors/random_sensors は TEMPLATE_ARGS で固定しているのでハードコードでよい)。
    return (
        Path(NAS_ROOT) / "output" / exp_name
        / "full_time_sensor_3d" / "k_36" / "36sensors_random" / "GPU0" / "log.txt"
    )


# main.py の log_average_results() が書き出す "--- Average Test Results (N runs) ---"
# ブロックを読む。log.csv 側はepoch毎の列しかヘッダーに無く、test_*系の列は
# ヘッダーに載らないまま値だけ追記される既存の実装上の癖があり
# csv.DictReaderでは正しく取れないため、log.txt のキー名付きテキストブロックを使う。
AVG_BLOCK_RE = re.compile(
    r"--- Average Test Results \((\d+) runs\) ---\s*"
    r"\{ 'Average Stats': \{\n(.*?)\n\} \}",
    re.DOTALL,
)
METRIC_LINE_RE = re.compile(r"'(\w+)':\s*([^,\n]+),?")


def read_avg_row(exp_name):
    path = output_log_txt(exp_name)
    if not path.exists():
        return None
    text = path.read_text()
    matches = list(AVG_BLOCK_RE.finditer(text))
    if not matches:
        return None
    n_runs_found, body = matches[-1].group(1), matches[-1].group(2)
    if int(n_runs_found) != N_RUNS:
        return None
    values = {}
    for m in METRIC_LINE_RE.finditer(body):
        key, val_str = m.group(1), m.group(2).strip()
        if key in METRICS:
            try:
                values[key] = float(val_str)
            except ValueError:
                return None
    if len(values) != len(METRICS):
        return None
    nan_metrics = [k for k, v in values.items() if v != v]  # NaN != NaN
    if nan_metrics:
        raise RuntimeError(f"{exp_name}: log.txt の指標に NaN が含まれています: {nan_metrics} -> {values}")
    return values


def validate_dataset_dirs(names):
    missing = []
    for name in names:
        d = NAS_DATA_ROOT / name
        for fname in ("light.csv", "bbox_label.csv", "environment.csv"):
            if not (d / fname).exists():
                missing.append(str(d / fname))
    if missing:
        raise FileNotFoundError(
            "以下のデータファイルが見つかりません:\n" + "\n".join(missing)
        )


def build_cmd(train_names, val_name, test_name, exp_name):
    return [
        sys.executable, "-u", str(MAIN_PY),
        "--train", *train_names,
        "--val", val_name,
        "--test", test_name,
        *TEMPLATE_ARGS,
        "--exp_name", exp_name,
    ]


def run_training(train_names, val_name, test_name, exp_name, cuda_visible_devices, dry_run):
    cmd = build_cmd(train_names, val_name, test_name, exp_name)
    print(f"[RUN] exp_name={exp_name}")
    print(f"      train={train_names}")
    print(f"      val={val_name} test={test_name} CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    if dry_run:
        print("      (dry-run: 実行しません)")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def ensure_result(train_names, val_name, test_name, exp_name, cuda_visible_devices, dry_run):
    result = read_avg_row(exp_name)
    if result is not None:
        print(f"[SKIP] {exp_name}: 既に {AVG_EPOCH_LABEL} の結果あり -> {result}")
        return result
    run_training(train_names, val_name, test_name, exp_name, cuda_visible_devices, dry_run)
    if dry_run:
        return None
    result = read_avg_row(exp_name)
    if result is None:
        raise RuntimeError(
            f"{exp_name}: 学習は完了したが log.txt に {AVG_EPOCH_LABEL} ブロックが見つかりません。"
        )
    return result


def decide_winner(candidate_metrics):
    """
    candidate_metrics: {case: {metric: value}}
    DECISION_METRICS を順に見て、上位が並んでいる候補集合を絞り込んでいく。
    最後まで複数残った場合は None を返し、呼び出し側で人間判断に回す。
    """
    remaining = list(candidate_metrics.keys())
    for metric in DECISION_METRICS:
        best_val = max(candidate_metrics[c][metric] for c in remaining)
        remaining = [c for c in remaining if candidate_metrics[c][metric] == best_val]
        if len(remaining) == 1:
            return remaining[0]
    return None if len(remaining) > 1 else remaining[0]


def load_state(state_path):
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"steps": {}, "finals": {}}


def save_state(state_path, state, dry_run):
    if dry_run:
        return
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def print_tie_table(size, candidate_metrics):
    print("=" * 78)
    print(f"[要人間判断] size={size}: {' -> '.join(DECISION_METRICS)} の順で見ても決着しません。")
    header = f"{'case':<28}" + "".join(f"{m:>14}" for m in METRICS)
    print(header)
    for c, m in candidate_metrics.items():
        print(f"{case_short(c):<28}" + "".join(f"{m[met]:>14.4f}" for met in METRICS))
    print("state_<axis>.json の該当ステップに \"winner\": \"<case>\" を手動で追記して再実行してください。")
    print("=" * 78)


def run_axis(axis, state_dir, dry_run):
    cases = AXIS_CASES[axis]
    cuda_visible_devices = AXIS_CUDA_VISIBLE_DEVICES[axis]
    state_path = state_dir / f"state_{axis}.json"
    state = load_state(state_path)

    validate_dataset_dirs([REAL_TRAIN, REAL_VAL, REAL_TEST, *cases])

    base = []
    remaining = list(cases)
    combo_by_size = {}

    for size in range(1, len(cases) + 1):
        step_key = str(size)
        step = state["steps"].setdefault(
            step_key, {"base": list(base), "candidates": {}, "winner": None}
        )

        if len(remaining) == 1:
            # 候補が1通りのみ(全ケース込み) -> 選定不要でそのまま採用
            if step["winner"] is None:
                step["winner"] = remaining[0]
                save_state(state_path, state, dry_run)
        elif step["winner"] is None:
            candidate_metrics = {}
            for case in remaining:
                trial_set = base + [case]
                exp_name = f"fwdsel_{axis}_sim03/step{size}_add_{case_short(case)}"
                metrics = ensure_result(
                    train_names=[REAL_TRAIN, *trial_set],
                    val_name=REAL_VAL,
                    test_name=REAL_VAL,  # 選定用: valの4指標を既存の平均化ロジックで取得
                    exp_name=exp_name,
                    cuda_visible_devices=cuda_visible_devices,
                    dry_run=dry_run,
                )
                if metrics is not None:
                    candidate_metrics[case] = metrics
                    step["candidates"][case] = metrics
                    save_state(state_path, state, dry_run)

            if dry_run:
                # dry-runでは実測値が無いため選定は行わない
                base = base + [remaining[0]]
                remaining = remaining[1:]
                combo_by_size[size] = list(base)
                continue

            winner = decide_winner(candidate_metrics)
            if winner is None:
                print_tie_table(size, candidate_metrics)
                save_state(state_path, state, dry_run)
                return
            step["winner"] = winner
            print(f"[DECIDED] axis={axis} size={size}: winner={case_short(winner)}")
            save_state(state_path, state, dry_run)

        winner = step["winner"]
        base = base + [winner]
        remaining = [c for c in remaining if c != winner]
        combo_by_size[size] = list(base)

    # 全サイズの選定完了 -> 各サイズのベスト組み合わせを本物のtestで最終確認
    for size, combo in combo_by_size.items():
        final_key = str(size)
        if state["finals"].get(final_key) is not None:
            continue
        exp_name = f"fwdsel_{axis}_sim03/final_size{size}"
        metrics = ensure_result(
            train_names=[REAL_TRAIN, *combo],
            val_name=REAL_VAL,
            test_name=REAL_TEST,  # 最終確認のみ本物のtest。選定には使わない
            exp_name=exp_name,
            cuda_visible_devices=cuda_visible_devices,
            dry_run=dry_run,
        )
        if metrics is not None:
            state["finals"][final_key] = {"combo": combo, "metrics": metrics}
            save_state(state_path, state, dry_run)

    if not dry_run:
        print_final_report(axis, state)


def print_final_report(axis, state):
    print("\n" + "=" * 78)
    print(f"=== axis {axis}: サイズ別ベスト組み合わせ 最終test結果 ===")
    header = f"{'size':>4} {'combo':<50}" + "".join(f"{m:>14}" for m in METRICS)
    print(header)
    for size_key in sorted(state["finals"], key=int):
        entry = state["finals"][size_key]
        combo_str = "+".join(case_short(c) for c in entry["combo"])
        m = entry["metrics"]
        print(f"{size_key:>4} {combo_str:<50}" + "".join(f"{m[met]:>14.4f}" for met in METRICS))
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", choices=["A", "B"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="コマンドとexp_nameを表示するだけで学習は実行しない")
    args = parser.parse_args()

    state_dir = Path(__file__).resolve().parent
    run_axis(args.axis, state_dir, args.dry_run)


if __name__ == "__main__":
    main()

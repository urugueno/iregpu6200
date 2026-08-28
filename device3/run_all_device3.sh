#!/bin/bash
# device3 配下の5本の実験を、前のスクリプトが完了してから次を実行する形で
# 順番に流すラッパースクリプト。
#
# 使い方:
#   nohup ./device3/run_all_device3.sh > ./device3/run_all_device3.log &
#
# 各スクリプトの標準出力は従来通りそれぞれの device3_*.txt に出力される。

cd "$(dirname "$0")/.."

echo "=== [1/5] device3.sh (real+sim, env none) ==="
bash ./device3/device3.sh > ./device3/device3.txt
echo "=== [1/5] done ==="

echo "=== [2/5] device3_real_only.sh (real only, env none) ==="
bash ./device3/device3_real_only.sh > ./device3/device3_real_only.txt
echo "=== [2/5] done ==="

echo "=== [3/5] device3_EQ.sh (env=query) ==="
bash ./device3/device3_EQ.sh > ./device3/device3_EQ.txt
echo "=== [3/5] done ==="

echo "=== [4/5] device3_EE.sh (env=PE) ==="
bash ./device3/device3_EE.sh > ./device3/device3_EE.txt
echo "=== [4/5] done ==="

echo "=== [5/5] device3_EE_EQ.sh (env=PE_query) ==="
bash ./device3/device3_EE_EQ.sh > ./device3/device3_EE_EQ.txt
echo "=== [5/5] done ==="

echo "=== All device3 experiments finished ==="

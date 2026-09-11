#!/bin/bash
# sim_num2 配下の4本の実験を、前のスクリプトが完了してから次を実行する形で
# 順番に流すラッパースクリプト。
#
# 使い方:
#   nohup ./basic_comp/sim_num2/run_all_sim_num2.sh &
#
# 各スクリプトの標準出力はそれぞれの sim_num2/*.txt に出力される。

cd "$(dirname "$0")/../.."

echo "=== [1/4] none.sh (env=none) ==="
bash ./basic_comp/sim_num2/none.sh > ./basic_comp/sim_num2/none.txt
echo "=== [1/4] done ==="

echo "=== [2/4] EE.sh (env=EE) ==="
bash ./basic_comp/sim_num2/EE.sh > ./basic_comp/sim_num2/EE.txt
echo "=== [2/4] done ==="

echo "=== [3/4] EQ.sh (env=EQ) ==="
bash ./basic_comp/sim_num2/EQ.sh > ./basic_comp/sim_num2/EQ.txt
echo "=== [3/4] done ==="

echo "=== [4/4] PE_query.sh (env=PE_query) ==="
bash ./basic_comp/sim_num2/PE_query.sh > ./basic_comp/sim_num2/PE_query.txt
echo "=== [4/4] done ==="

echo "=== All sim_num2 experiments finished ==="

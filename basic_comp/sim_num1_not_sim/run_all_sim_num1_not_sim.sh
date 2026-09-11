#!/bin/bash
# sim_num1_not_sim 配下の4本の実験を、前のスクリプトが完了してから次を実行する形で
# 順番に流すラッパースクリプト。
#
# 使い方:
#   nohup ./basic_comp/sim_num1_not_sim/run_all_sim_num1_not_sim.sh &
#
# 各スクリプトの標準出力はそれぞれの sim_num1_not_sim/*.txt に出力される。

cd "$(dirname "$0")/../.."

echo "=== [1/4] sim_EE.sh (env=EE) ==="
bash ./basic_comp/sim_num1_not_sim/sim_EE.sh > ./basic_comp/sim_num1_not_sim/sim_EE_ver1.txt
echo "=== [1/4] done ==="

echo "=== [2/4] sim_EQ.sh (env=EQ) ==="
bash ./basic_comp/sim_num1_not_sim/sim_EQ.sh > ./basic_comp/sim_num1_not_sim/sim_EQ_ver1.txt
echo "=== [2/4] done ==="

echo "=== [3/4] sim_PE_query.sh (env=PE_query) ==="
bash ./basic_comp/sim_num1_not_sim/sim_PE_query.sh > ./basic_comp/sim_num1_not_sim/sim_PE_query_ver1.txt
echo "=== [3/4] done ==="

# echo "=== [4/4] not_sim.sh (env=none, no sim data) ==="
# bash ./basic_comp/sim_num1_not_sim/not_sim.sh > ./basic_comp/sim_num1_not_sim/not_sim.txt
# echo "=== [4/4] done ==="

echo "=== All sim_num1_not_sim experiments finished ==="

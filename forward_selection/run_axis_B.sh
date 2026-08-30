#!/bin/bash
# 軸B(光源間隔, 8ケース)の逐次前向き選択を実行する。GPU1を使用。
# 途中で止まっても再実行すれば state_B.json を見て続きから進む。
#
# 使い方:
#   nohup ./forward_selection/run_axis_B.sh > ./forward_selection/run_axis_B.log &

cd "$(dirname "$0")/.."
python3 -u forward_selection/run_forward_selection.py --axis B

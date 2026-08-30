#!/bin/bash
# 軸A(光源数, 6ケース)の逐次前向き選択を実行する。GPU0を使用。
# 途中で止まっても再実行すれば state_A.json を見て続きから進む。
#
# 使い方:
#   nohup ./forward_selection/run_axis_A.sh > ./forward_selection/run_axis_A.log &

cd "$(dirname "$0")/.."
python3 -u forward_selection/run_forward_selection.py --axis A

CUDA_VISIBLE_DEVICES=0 python -u Illuminance_DETR/main.py \
    --train keita_2026_0617/1_tsunoda_1_train  sim_test_01/loop1 sim_test_04/loop1\
    --val   keita_2026_0617/1_tsunoda_1_val \
    --test  keita_2026_0617/1_tsunoda_1_test \
    --exp_name sim_num2_EQ \
    --task bbox --num_classes 4 \
    --batch_size 256 --epochs 1000 --early_stopping_patience 50 --n_runs 3 \
    --model_mode time_sensor --scale full --environment EQ \
    --num_sensors 36 --k_neighbors 36

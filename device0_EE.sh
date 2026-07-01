CUDA_VISIBLE_DEVICES=0 python -u Illuminance_bbox/main.py \
    --data sim_lecture_02/loop1 sim_lecture_02/loop2 sim_lecture_02/loop3 \
    --batch_size 256 --epochs 1000000 --dataset_file illuminance --backbone illuminance \
    --early_stopping_patience 50 --standardize --n_runs 3 \
    --window_size 41 --sub_window_size 37 --stride 1 --scale full --model_mode time_sensor \
    --num_queries 10 --enc_layers 3 --dec_layers 3 --num_classes 4 --environment PE \
    --num_sensors 36 --k_neighbors 36 --exp_name tmp_EE --sensor_dropout_rate 0.1 --random_sensors;

# nohup ./device0_EE.sh > ./device0_EE.txt &
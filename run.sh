export HF_ENDPOINT=https://hf-mirror.com

nohup python train.py --label_columns 2 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:0 --alpha 0.75 --gamma 1 > ./loginfo/logfile_2.out 2>&1 &
nohup python train.py --label_columns 3 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:1 --alpha 0.75 --gamma 1 > ./loginfo/logfile_3.out 2>&1 &
nohup python train.py --label_columns 4 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:2 --alpha 0.75 --gamma 1 > ./loginfo/logfile_4.out 2>&1 &
nohup python train.py --label_columns 5 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:3 --alpha 0.75 --gamma 1 > ./loginfo/logfile_5.out 2>&1 &
nohup python train.py --label_columns 6 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:4 --alpha 0.75 --gamma 1 > ./loginfo/logfile_6.out 2>&1 &
nohup python train.py --label_columns 7 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:5 --alpha 0.75 --gamma 1 > ./loginfo/logfile_7.out 2>&1 &
nohup python train.py --label_columns 8 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:2 --alpha 0.75 --gamma 1 > ./loginfo/logfile_8.out 2>&1 &
nohup python train.py --label_columns 9 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:3 --alpha 0.75 --gamma 1 > ./loginfo/logfile_9.out 2>&1 &
nohup python train.py --label_columns 10 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:4 --alpha 0.75 --gamma 1 > ./loginfo/logfile_10.out 2>&1 &
nohup python train.py --label_columns 11 --batch_size 128 --num_epochs 20 --lr 2e-5 --eval_rounds 1 --assign_gate 0.5 --device cuda:5 --alpha 0.75 --gamma 1 > ./loginfo/logfile_11.out 2>&1 &
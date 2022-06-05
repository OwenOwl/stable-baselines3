CUDA_VISIBLE_DEVICES=2 python3 -u main/train_relocate_pc_img.py --exp lr_0.0001_bs_512 --n 100 --workers 10  --iter 5000 --object_name mustard_bottle --lr 0.0001 --seed 0 --bs 512 --img_type goal_robot

CUDA_VISIBLE_DEVICES=0 python3 -u main/train_relocate_pc_img.py --exp lr_0.0003_bs_512 --n 100 --workers 6  --iter 5000 --object_name mustard_bottle --lr 0.0003 --seed 0 --bs 512 --img_type goal_robot

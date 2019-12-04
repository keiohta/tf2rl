#/usr/bin/bash

python run_ddpg_mujoco.py --env-name HalfCheetah-v2 --gpu -1 &
python run_ddpg_mujoco.py --env-name Hopper-v2 --gpu -1 &
python run_ddpg_mujoco.py --env-name Walker2d-v2 --gpu -1 &
python run_ddpg_mujoco.py --env-name Ant-v2 --gpu -1 &
python run_ddpg_mujoco.py --env-name Reacher-v2 --gpu -1 &
python run_ddpg_mujoco.py --env-name InvertedPendulum-v2 --gpu -1 &
python run_ddpg_mujoco.py --env-name InvertedDoublePendulum-v2 --gpu -1
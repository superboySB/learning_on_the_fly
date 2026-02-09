# learning_on_the_fly Docker 速记

## 一步到位（默认 8889 代理）
`docker/Dockerfile` 已内置：
- `Ubuntu 20.04`
- `CUDA 12.9`
- `ROS Noetic (ROS1, 含 rospy)`
- 默认 `python3=3.9`（无 conda/micromamba）
- JAX 按 CUDA 方式安装

```bash
cd /home/dzp/projects/learning_on_the_fly

docker build -f docker/Dockerfile \
  -t dzp_lotf:cuda129-u2004-noetic-py39 \
  --network=host --progress=plain .

xhost +local:root

docker run --name dzp-lotf -itd --privileged --gpus all --network host \
  --entrypoint bash \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  -e http_proxy=http://127.0.0.1:8889 \
  -e https_proxy=http://127.0.0.1:8889 \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --shm-size=4g \
  -v /home/dzp/projects/learning_on_the_fly:/workspace/learning_on_the_fly \
  dzp_lotf:cuda129-u2004-noetic-py39

docker exec -it dzp-lotf /bin/bash

cd /workspace/learning_on_the_fly
```

## 代码用法（按 README）
```bash
source /opt/ros/noetic/setup.bash
python3 -c "import rospy, jax; print('rospy+jax ok')"
python3 -m pip install -e .
```

从 `README.md` 推荐入口：
- `examples/residual_dynamics/train_ensemble_model.ipynb`
- `examples/state_hovering/`
- `examples/traj_tracking/`
- `examples/vision_hovering/`

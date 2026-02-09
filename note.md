# learning_on_the_fly Docker 速记

## 配置
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

## 技术笔记
- 本项目关注“可微仿真 + 残差动力学 + 策略快速适配”的组合范式，核心目标是用可微动力学做 BPTT 更新策略，同时用残差模型补偿未建模扰动，从而缩小 sim-to-real gap。
- “可微”指的是仿真步进对状态/动作可微，以便对策略参数做梯度更新；并非在线估计整机物理参数。基础动力学参数（质量、惯量、推力曲线等）来自固定机体配置，残差模型学习的是“模型误差/扰动”而不是通用动力学。
- `examples/residual_dynamics/example_dataset.csv` 是离线残差数据集，形状 1000x22，输入 19 维、输出 3 维。输入顺序与 `lotf/objects/quadrotor_obj.py` 中 `state_for_res` 一致：`p(3)`, `R(9)`, `v(3)`, `f_d(1)`, `omega_d(3)`，输出为 `res_acc(3)`，属于监督学习标签；仓库未提供该数据集的生成流程。
- `res_acc` 是世界系线加速度残差，简化动力学公式为 `dv/dt = gravity + R@[0,0,a] + res_acc`（`a = f_d / mass`，`R` 是 body→world 旋转矩阵）。默认实现把残差加在世界系加速度上，若需要机体系残差可通过 `R^T` 做变换并统一数据约定。
- “模型预测加速度”来自基础动力学（简化或高保真），残差定义为“真实加速度 − 基础模型加速度”，因此残差模型与基础模型耦合，跨机体/跨模型不天然通用，需要重新训练或在线更新。该范式本质是“模型误差补偿”，而非无模型动力学。
- 策略输出为 `[f_d, ωx, ωy, ωz]`（总推力 + 期望机体角速度），不是电机指令。`use_high_fidelity=True` 时仿真内部会通过 Betaflight 风格控制器把指令转为电机转速；`use_high_fidelity=False` 时直接进入简化动力学。
- `use_forward_residual=True` 仅影响前向仿真轨迹，当前实现的 BPTT 反向梯度只通过简化动力学，不包含残差模型梯度（`Quadrotor.step` 的自定义 JVP）。因此残差影响前向但不参与反向链路。
- `1_train_base_policy` 从头训练基础策略；`3_finetune_policy_full` 在已有策略上全量微调；`3_finetune_policy_lora` 仅更新 LoRA 低秩参数、基座权重冻结。是否启用 residual 不改变“能否微调”，但启用更贴近论文的快速适配叙事。
- 视觉悬停多出的 `examples/vision_hovering/1_pretrain_base_policy.ipynb` 是为了解决“视觉特征→控制”难以直接收敛的问题：先用 `HoveringFeaturesEnv` 采集 rollout，把视觉特征当输入、把真实状态当监督标签训练“状态预测器”MLP（L1 损失），再把预测器的前两层 `Dense_0/Dense_1` 权重拷贝到策略网络作为初始化，保存为 `vision_hovering_pre_params`，之后 `2_train_base_policy.ipynb` 再在悬停任务上继续训练。输入维度来自 `HoveringFeaturesEnv.observation_space`，默认配置 `num_last_quad_states=15`、`skip_frames=3`、`delay=0.04`、`dt=0.02` 时 `num_frames=5`、`num_last_actions=3`，因此输入维度为 `2*7*num_frames + 4*num_last_actions = 82`（5 帧、每帧 7 个点的 2D 投影 + 3 步动作历史）。监督标签是当前真实状态的拼接：`p(3)` + `R(9)` + `v(3)`，总计 15 维。
- 论文强调“在线交替学习”（真实 rollout → 残差监督更新 → BPTT 微调策略 → 再 rollout），而当前 repo 只提供离线残差训练 + 离线策略微调的分块示例，完整闭环需自行实现数据采集与在线更新。
- 对未知复杂环境（如矿洞避障）而言，主要缺口在感知、建图、避障奖励与闭环数据流；残差动力学只能修正动力学误差，不能替代环境理解与决策模块。

# 代码结构
- `lotf/` 是核心库：`algos/` 实现 BPTT 训练（`lotf/algos/bptt.py`），`envs/` 是悬停/轨迹跟踪环境（`hovering_state_env.py`, `traj_tracking_state_env.py`, `hovering_features_env.py`），`modules/` 包含策略 MLP 与残差 MLP，`objects/` 包含 `Quadrotor` 和参考轨迹对象，`sensors/` 目前只有简化相机模型，`simulation/` 提供高保真动力学与旋翼残差模型，`utils/` 放残差训练与工具函数。
- 关键超参主要出现在 notebook：`sim_dyn_config` 控制仿真模式（`use_high_fidelity`, `use_forward_residual`），`num_envs/max_epochs/sim_dt/max_sim_time` 控制训练规模与 rollout 长度，环境里的 `*_std` 与 reward 权重控制噪声与优化目标。这些只影响训练过程，不改变“学习对象”（base/finetune/LoRA 才改变学习对象）。
- `examples/residual_dynamics/train_ensemble_model.ipynb` 用 `example_dataset.csv` 训练残差 MLP 集成（默认 3 个模型），输出保存到 `checkpoints/residual_dynamics/my_residual_dynamics_params`；它是纯监督学习，与策略无关。
- `examples/state_hovering/`、`examples/traj_tracking/` 结构一致：`1_train_base_policy.ipynb` 从头训练策略，`2_eval_policy.ipynb` 评估，`3_finetune_policy_full.ipynb` 全量微调，`3_finetune_policy_lora.ipynb` 低秩微调。`examples/vision_hovering/` 多一步 `1_pretrain_base_policy.ipynb`：先在 `HoveringFeaturesEnv` 里用 rollouts 训练“视觉特征→状态”的预测器，再把前两层权重拷贝到策略网络保存为 `vision_hovering_pre_params`，用于 `2_train_base_policy.ipynb` 的悬停训练初始化。
- `checkpoints/` 使用 Orbax OCDBT 格式保存参数。`policy/` 存策略参数（如 `state_hovering_params`、`traj_tracking_params`、`vision_hovering_params`），`residual_dynamics/` 存残差参数（如 `dummy_params`、`example_params`、`my_residual_dynamics_params`）。内部的 `_METADATA/_sharding/manifest.ocdbt/ocdbt.process_0` 是 Orbax 的索引与分片文件，不是“截图/日志”，直接用 `PyTreeCheckpointer().restore()` 读取即可。

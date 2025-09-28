# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass #装饰器将普通类转换为配置类，启用配置继承和组合
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    # observation_space = 9
    observation_space = 3
    state_space = 0
    
    # simulation 
    sim: SimulationCfg = SimulationCfg(
    dt=1/120,                    # 8.33ms的物理步长
    render_interval=2,           # 每隔一帧渲染
    gravity=(0.0, 0.0, -9.81),  # 重力向量
    physics_material=...,        # 物理材质属性
    use_gpu_pipeline=True,       # 使用GPU加速
    )

    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=4096, #并行环境数量
    env_spacing=4.0, #环境间距(米)
    replicate_physics=True
    )
    dof_names = ["left_wheel_joint", "right_wheel_joint"]
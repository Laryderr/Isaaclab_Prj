# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils #仿真工具函数
from isaaclab.assets import Articulation #关节体机器人类
from isaaclab.envs import DirectRLEnv #直接强化学习环境基类
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane #地面生成函数
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg #自己的环境配置类

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg #可视化标记类
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR #Isaac Nucleus 资源路径
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg( 
        prim_path="/Visuals/myMarkers", 
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg #类型注解，指定配置类类型

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #从机器人获取关节索引
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        # setting aside useful variables for later
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.commands[:,-1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)

        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        

    def _visualize_markers(self):
        # get marker locations and orientations
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # offset markers so they are above the jetbot
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    # root_com_lin_vel_b 是 ArticulationData 的一个属性，处理了质心线性速度从世界坐标系到本体坐标系的转换。
    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w #质心线速度，世界坐标系

        # quat_apply(quaternion, vector) 的作用：将向量从一个坐标系旋转到另一个坐标系
        # FORWARD_VEC_B = [1, 0, 0]  (本体坐标系中的前进方向)
        # root_link_quat_w: 机器人相对于世界坐标系的旋转四元数
        # 结果: 机器人在世界坐标系中的前进方向向量
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B) 

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        obs = torch.hstack((dot, cross, forward_speed))

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # root_com_lin_vel_b: 本体坐标系下的线性速度
        # 形状: [num_envs, 3] = [[vx, vy, vz], [vx, vy, vz], ...]

        # [:,0] 取第0列（x方向速度）
        # 形状: [num_envs] = [vx, vx, vx, ...]

        # .reshape(-1,1) 转为列向量
        # 形状: [num_envs, 1] = [[vx], [vx], [vx], ...]
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1) #前进移动奖励
        # 向量点积公式：A·B = |A||B|cos(θ)
        # 当A和B都是单位向量时：A·B = cos(θ)
        # θ是两向量间夹角

        # self.forwards: [num_envs, 3]  机器人前进方向
        # self.commands: [num_envs, 3]  期望命令方向

        # 逐元素乘法：
        # self.forwards * self.commands = [[fx*cx, fy*cy, fz*cz], ...]

        # torch.sum(..., dim=-1): 沿最后维度求和
        # 结果：[fx*cx + fy*cy + fz*cz, ...] = 点积结果

        # keepdim=True: 保持形状为 [num_envs, 1] 而不是 [num_envs]
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True) #对齐奖励
        total_reward = forward_reward*torch.exp(alignment_reward) #用指数函数将对其奖励全部映射到正值，当我们不对齐时就不会得到奖励，
                                                                  #且不会出现对其奖励和前进奖励都为负数乘积为正的退化解
        return total_reward

    # 标记哪些环境需要重置以及原因
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: #包含两个指定类型的元组 
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out #第一个元素表示是否因为失败而终止，第二个表示是否因为时间用尽而终止

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # pick new commands for reset envs
        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)

        # recalculate the orientations for the command markers with the new commands
        ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids]< 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        # set the root state for the reset envs
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()

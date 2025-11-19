# 导入系统相关库
import sys
import os
# 将上级目录添加到系统路径，便于导入自定义模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 导入正则表达式、进度条、PyTorch等工具库
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np

# 导入类型注解、配置工具、图像处理等库
from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile, ImageDraw, ImageFont
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
# 导入深度图像过滤函数
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array

# 导入Habitat环境相关库（用于导航模拟）
import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

# 导入自定义模型和工具函数
from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import dict_to_cuda
from utils.dist import *  # 分布式处理工具
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN


class VLNEvaluator:
    """视觉语言导航(VLN)评估器类，用于评估模型在 Habitat 环境中的导航性能"""
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        tokenizer: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args  # 命令行参数
        self.device = torch.device('cuda')  # 使用GPU设备
        self.split = split  # 评估数据集分割（如val_seen/val_unseen）
        self.env_num = env_num  # 环境数量
        self.save_video = args.save_video  # 是否保存导航视频
        self.output_path = output_path  # 结果输出路径
        self.epoch = epoch  # 当前训练轮次
        self.config_path = config_path  # Habitat配置文件路径
        self.config = get_habitat_config(config_path)  # 加载Habitat配置
        self.agent_config = get_agent_config(self.config.habitat.simulator)  # 智能体配置
        # 获取传感器配置（RGB和深度传感器）
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        # 修改Habitat配置（读写模式）
        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split  # 设置数据集分割
            # 更新任务测量指标（添加顶视图地图和碰撞检测）
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,  # 地图填充
                        map_resolution=1024,  # 地图分辨率
                        draw_source=True,  # 绘制起点
                        draw_border=True,  # 绘制边界
                        draw_shortest_path=True,  # 绘制最短路径
                        draw_view_points=True,  # 绘制视角点
                        draw_goal_positions=True,  # 绘制目标位置
                        draw_goal_aabbs=True,  # 绘制目标包围盒
                        fog_of_war=FogOfWarConfig(  # 战争迷雾配置
                            draw=True,
                            visibility_dist=5.0,  # 可见距离
                            fov=90,  # 视野角
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),  # 碰撞测量配置
                }
            )

        print(f"config类型 = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))  # 打印配置信息

        # 提取相机参数
        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]  # 相机高度
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth  # 最小深度
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth  # 最大深度

        # 计算相机内参（视场角转弧度，计算fx/fy）
        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        
        # 初始化图像处理器、模型和分词器
        self.image_processor = model.get_vision_tower().image_processor
        self.model = model
        self.tokenizer = tokenizer
        
        # 初始化对话提示模板（指导模型生成导航动作）
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        
        # 动作到索引的映射（STOP/前进/左转/右转）
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })
        
        # 描述观测的随机前缀（增加输入多样性）
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        
        # 模型相关参数（帧数量、未来步骤数、历史记录数）
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history
    
    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        """预处理深度图像：调整尺寸并缩放深度值"""
        # 目标尺寸（与RGB图像处理器一致）
        target_height = self.image_processor.crop_size['height']
        target_width = self.image_processor.crop_size['width']
        # 调整深度图像大小（最近邻插值）
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        
        # 转换为数组并缩放（单位转换，如米转毫米）
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale  # 缩放深度值
        
        return img, (target_width, target_height)
    
    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        """计算相机内参矩阵（针孔相机模型）"""
        width = sensor_cfg.width  # 图像宽度
        height = sensor_cfg.height  # 图像高度
        fov = sensor_cfg.hfov  # 水平视场角
        # 计算焦距fx/fy（像素单位）
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # 假设正方形像素，fx=fy
        # 主点坐标（图像中心）
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        # 4x4内参矩阵（齐次坐标）
        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):
        """预处理内参矩阵：根据图像尺寸调整（缩放和裁剪）"""
        intrinsic = copy.deepcopy(intrinsic)
        # 扩展维度（适配批量处理）
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # 增加批次维度
        
        # 根据图像尺寸缩放内参（宽度和高度方向）
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # 宽度方向缩放
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # 高度方向缩放

        # 处理裁剪变换（调整主点坐标）
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        # 移除多余维度（若批次为1）
        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def get_axis_align_matrix(self):
        """获取轴对齐矩阵（转换 Habitat 坐标系到标准坐标系）"""
        # 转换矩阵：Habitat坐标系 -> 标准相机坐标系
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
        return ma
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        """将位置（xyz）和偏航角（yaw）转换为4x4变换矩阵"""
        x, y, z = xyz
        # 旋转+平移矩阵（绕z轴旋转yaw角）
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],  # 旋转+平移（x）
                [np.sin(yaw), np.cos(yaw), 0, y],  # 旋转+平移（y）
                [0, 0, 1, z],  # 平移（z）
                [0, 0, 0, 1],  # 齐次坐标
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        """配置并初始化Habitat环境"""
        env = Env(config=self.config)
        # 可注释掉：仅用于调试单episode
        # env.episodes = env.episodes[0:1]
        return env

    def eval_action(self, idx) -> None:
        """核心评估函数：遍历场景和episode，执行导航并记录指标"""
        env = self.config_env()  # 初始化环境
        # 按场景分组episode（减少场景切换开销）
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        # 获取RGB传感器内参矩阵
        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        # 存储评估指标：成功率、SPL、 oracle成功率、到目标距离
        sucs, spls, oss, ones = [], [], [], []
        done_res = []  # 已完成的episode记录（避免重复评估）
        
        # 加载已完成的结果（若存在）
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:  # 主进程记录指标
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        
        # 遍历每个场景
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]  # 提取场景ID
            print(f"当前场景ID = {scene_id}")
            # 进度条：处理当前进程负责的episode（分布式拆分）
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"场景 {scene_id}")
            # 遍历当前进程负责的episode
            for episode in episodes[idx::self.env_num]:
                # 获取导航指令（普通导航为文本，目标导航为物体类别）
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("开始episode：", episode_instruction)
                episode_id = episode.episode_id
                # 跳过已完成的episode
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                
                # 重置模型状态（针对当前环境）
                self.model.reset_for_env(idx)
                # 设置当前episode并重置环境
                env.current_episode = episode
                observations = env.reset()
                # 保存初始RGB图像（调试用）
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                
                vis_frames = []  # 可视化帧（用于生成视频）
                step_id = 0  # 步数计数器
                
                # 若保存视频，创建目录
                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)
                # 记录初始高度（用于计算相对高度）
                initial_height = env.sim.get_agent_state().position[1]

                # 存储序列数据：RGB、深度、位姿、内参、时间ID、动作序列
                rgb_list = []
                depth_list = []
                depth_images_list = []
                pose_list = []
                intrinsic_list = []
                time_ids = []
                action_seq = []
                past_key_values = None  # 模型缓存（用于增量生成）
                output_ids = None  # 模型输出ID
                
                # 导航循环：直到episode结束
                while not env.episode_over:
                    self.model.eval()  # 模型设为评估模式
                    time_ids.append(step_id)  # 记录当前步数
                    
                    # 获取当前观测
                    rgb = observations["rgb"]  # RGB图像
                    depth = observations["depth"]  # 深度图像
                    x, y = observations["gps"]  # GPS坐标
                    camera_yaw = observations["compass"][0]  # 指南针（偏航角）
                    
                    # 过滤深度图像（去噪）
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    # 深度值映射到[min_depth, max_depth]并转换为毫米
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    # 获取智能体状态（位置和旋转）
                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height  # 相对初始高度
                    # 相机位置（GPS坐标转换）
                    camera_position = np.array([x, -y, self._camera_height + height])
                    robot_xy = camera_position[:2]
                    # 相机到位姿坐标系的变换矩阵
                    tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                    
                    # 从四元数获取旋转矩阵，构建变换矩阵
                    rotation = agent_state.rotation
                    translation = agent_state.position
                    rotation_matrix = quaternion.as_rotation_matrix(rotation)
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = translation
                    
                    # 预处理RGB图像（适配模型输入）
                    image = Image.fromarray(rgb).convert('RGB')
                    image_size = image.size
                    image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                    
                    # 预处理深度图像
                    depth_image, resize_shape = self.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)
                    
                    # 预处理内参矩阵（适配图像尺寸）
                    intrinsic = self.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                    intrinsic = torch.from_numpy(intrinsic).float()
    
                    # 存储当前帧数据
                    rgb_list.append(image)
                    depth_list.append(torch.from_numpy(depth_image).float())
                    pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())
                    intrinsic_list.append(intrinsic)
                    
                    # 导航指令（普通导航为文本，目标导航为物体类别）
                    episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                    # print("## episode_instruction:", episode_instruction)

                    # 获取当前指标并可视化
                    info = env.get_metrics()
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)  # 生成可视化帧
                        # vis_frames.append(frame)

                        frame_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(frame_pil)
                        img_width, img_height = frame_pil.size

                        # --------------------------
                        # 1. 配置文字参数（左侧布局，增大字体）
                        # --------------------------
                        # 任务目标文字
                        task_text = f"## Task: {episode_instruction}"
                        # 导航结果信息（实时获取当前指标）
                        metrics = env.get_metrics()  # 实时获取当前步骤的指标
                        result_text = (
                            f"scene_episode ID: {scene_id}_{episode_id}\n"
                            # f"success={metrics['success']:.1f}\n"
                            # f"SPL={metrics['spl']:.1f}\n"
                            f"oracle_success = {metrics['oracle_success']:.1f}\n"
                            f"distance_to_goal = {metrics['distance_to_goal']:.2f}"
                        )
                        # 合并文本（任务目标 + 导航结果，用空行分隔）
                        full_text = f"{task_text}\n{result_text}"

                        # 字体和布局参数
                        base_font_size = int(img_height * 0.04)  # 字体大小：图像高度的4%
                        margin = int(img_height * 0.015)  # 边距
                        line_spacing = int(base_font_size * 0.3)  # 行间距
                        text_color = (255, 255, 255)  # 白色文字
                        bg_color = (0, 0, 0, 200)  # 深色半透明背景（提高可读性）
                        
                        # 加载字体（优先支持中文）
                        try:
                            font = ImageFont.truetype("simhei.ttf", base_font_size)
                        except:
                            font = ImageFont.load_default()


                        # --------------------------
                        # 2. 自动换行（限制宽度为图片的2/5，左侧布局）
                        # --------------------------
                        max_line_width = int(img_width * 0.4)  # 宽度不超过图片的2/5

                        def wrap_text(text, font, max_width):
                            """将文本按最大宽度拆分为多行"""
                            lines = []
                            current_line = ""
                            for char in text:
                                # 处理换行符（保留用户手动换行）
                                if char == "\n":
                                    lines.append(current_line)
                                    current_line = ""
                                    continue
                                # 测试当前行宽度
                                test_line = current_line + char
                                bbox = draw.textbbox((0, 0), test_line, font=font)
                                if (bbox[2] - bbox[0]) > max_width:
                                    lines.append(current_line)
                                    current_line = char
                                else:
                                    current_line = test_line
                            if current_line:
                                lines.append(current_line)
                            return lines

                        # 拆分文本为多行（保留手动换行）
                        wrapped_lines = wrap_text(full_text, font, max_line_width)


                        # --------------------------
                        # 3. 计算文本区域大小（左侧定位）
                        # --------------------------
                        # 单行高度 = 字体大小 + 行间距
                        line_height = base_font_size + line_spacing
                        # 总高度 = 行数×单行高度 - 最后一行的行间距
                        total_text_height = (len(wrapped_lines) * line_height) - line_spacing
                        # 最大单行宽度（取最长行的宽度）
                        max_line_width_actual = 0
                        for line in wrapped_lines:
                            bbox = draw.textbbox((0, 0), line, font=font)
                            line_width = bbox[2] - bbox[0]
                            if line_width > max_line_width_actual:
                                max_line_width_actual = line_width


                        # --------------------------
                        # 4. 绘制左侧背景框
                        # --------------------------
                        # 左侧定位：x从margin开始，y从margin开始
                        bg_x1 = margin
                        bg_y1 = margin
                        # 背景框宽度：最大单行宽度 + 2×边距（确保包裹文字）
                        bg_x2 = bg_x1 + max_line_width_actual + 2 * margin
                        # 背景框高度：总文本高度 + 2×边距
                        bg_y2 = bg_y1 + total_text_height + 2 * margin
                        # 确保背景框不超出图像边界（底部和右侧）
                        if bg_y2 > img_height - margin:
                            bg_y2 = img_height - margin  # 底部不超过图像边缘
                        if bg_x2 > img_width * 0.4 + margin:  # 右侧不超过2/5宽度
                            bg_x2 = int(img_width * 0.4) + margin
                        # 绘制背景框
                        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)


                        # --------------------------
                        # 5. 绘制左侧多行文字
                        # --------------------------
                        text_x = bg_x1 + margin  # 左对齐，留边距
                        text_y = bg_y1 + margin  # 上对齐，留边距
                        for line in wrapped_lines:
                            # 确保文字不超出背景框底部
                            if text_y + base_font_size > bg_y2 - margin:
                                break  # 超出则停止绘制（避免溢出）
                            draw.text((text_x, text_y), line, font=font, fill=text_color)
                            text_y += line_height  # 下移到下一行


                        # --------------------------
                        # 6. 转回numpy数组，添加到帧列表
                        # --------------------------
                        frame = np.array(frame_pil)
                        vis_frames.append(frame)
                                        


                    # 若动作序列为空，调用模型生成动作
                    if len(action_seq) == 0:
                        if output_ids is None:
                            # 初始化对话模板（首次生成）
                            sources = copy.deepcopy(self.conversation)
                            # 替换模板中的指令占位符
                            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                            # 若不是第一步，添加历史观测标记
                            if step_id != 0 :
                                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                            # 清理模板中的冗余标记
                            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', episode.instruction.instruction_text)
                            add_system = True  # 添加系统提示
                            print(step_id, sources[0]["value"])  # 打印输入提示
                        else:
                            # 增量生成：复用历史对话
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                            
                        # 预处理文本输入（转换为token ID）
                        input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                        # 增量拼接历史输出
                        if output_ids is not None:
                            input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)

                        # 选择输入帧（当前帧+历史帧，根据配置）
                        images = rgb_list[-1:]
                        depths = depth_list[-1:]
                        poses = pose_list[-1:]
                        intrinsics = intrinsic_list[-1:]
                        # 若达到指定步数，添加历史帧
                        if step_id != 0 and step_id % self.num_frames == 0:
                            if self.num_history is None:
                                history_ids = slice(0, time_ids[0], self.num_future_steps)
                            else:
                                history_ids = slice(0, time_ids[0], (time_ids[0] // self.num_history))
                            images = rgb_list[history_ids] + images
                            depths = depth_list[history_ids] + depths
                            poses = pose_list[history_ids] + poses
                            intrinsics = intrinsic_list[history_ids] + intrinsics
                                
                        # 构建模型输入字典（图像、深度、位姿、内参、文本等）
                        input_dict = {
                            'images': torch.stack(images).unsqueeze(0),
                            'depths': torch.stack(depths).unsqueeze(0),
                            'poses': torch.stack(poses).unsqueeze(0),
                            'intrinsics': torch.stack(intrinsics).unsqueeze(0),
                            'inputs': input_ids,
                            'env_id': idx,
                            'time_ids': [time_ids],
                            'task_type': [0]
                        }
                            
                        # 输入转移到GPU
                        input_dict = dict_to_cuda(input_dict, self.device)
                        
                        # 转换为bfloat16精度（加速推理）
                        for key, value in input_dict.items():
                            if key in ['images', 'depths', 'poses', 'intrinsics']:
                                input_dict[key] = input_dict[key].to(torch.bfloat16)
                        
                        # 模型生成动作序列
                        outputs = self.model.generate(
                            **input_dict,
                            do_sample=False,  # 确定性生成
                            num_beams=1,  #  beam search大小
                            max_new_tokens=10000,  # 最大生成token数
                            use_cache=True,  # 使用缓存加速
                            return_dict_in_generate=True,
                            past_key_values=past_key_values  # 复用历史缓存
                        )
                        
                        # 更新输出和缓存
                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values
                        # 解码生成结果
                        llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        print(llm_outputs, flush=True)  # 打印模型输出
                        # 解析动作序列
                        action_seq = self.parse_actions(llm_outputs)
                        print('解析的动作序列', action_seq, flush=True)
                        # 若未解析到动作，默认停止
                        if len(action_seq) == 0:
                            action_seq = [0]
                    
                    # 执行下一个动作
                    action = action_seq.pop(0)
                    observations = env.step(action)  # 环境执行动作并返回新观测
                    step_id += 1  # 步数+1
                    
                    # 每num_frames步重置模型状态（避免缓存过大）
                    if step_id % self.num_frames == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []
                
                process_bar.update(1)  # 更新进度条
                # 获取当前episode的评估指标
                metrics = env.get_metrics()
                # 保存视频（若配置）
                if self.save_video:
                    images_to_video(
                        vis_frames, 
                        os.path.join(self.output_path, f'vis_{self.epoch}'), 
                        f'{scene_id}_{episode_id}', 
                        fps=6, 
                        quality=9
                    )
                vis_frames.clear()  # 清空可视化帧
                # 记录指标
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"场景-episode {scene_id}_{episode_id} 结果：成功={metrics['success']}, SPL={metrics['spl']}, Oracle成功={metrics['oracle_success']}, 到目标距离={metrics['distance_to_goal']}")
                
                # 保存当前episode结果
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()  # 关闭环境
        # 返回当前进程的指标（转换为tensor便于分布式汇总）
        return (torch.tensor(sucs).to(self.device), 
                torch.tensor(spls).to(self.device), 
                torch.tensor(oss).to(self.device), 
                torch.tensor(ones).to(self.device), 
                torch.tensor(len(sucs)).to(self.device))     

    def parse_actions(self, output):
        """解析模型输出文本，提取动作序列"""
        # 构建动作匹配模式（正则表达式）
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)  # 匹配动作
        # 转换为动作索引
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)  # 展平列表
        return list(actions)

    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.", add_system: bool = False):
        """预处理Qwen模型的输入文本：转换为token并添加图像/记忆标记"""
        # 角色映射（human->user, gpt->assistant）
        roles = {"human": "user", "gpt": "assistant"}
        
        # 深拷贝分词器，避免修改原对象
        tokenizer = copy.deepcopy(tokenizer)
        # 添加图像和记忆特殊token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        # 获取特殊token的ID
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        # 无需mask的token索引
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids  # 换行符的token

        # 重置Qwen的对话模板（避免重复添加系统提示）
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # 处理输入文本，转换为模型可接受的格式
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            # 随机选择前缀，拼接图像标记
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            # 更新用户输入（添加图像提示）
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            # 跳过非人类的起始对话
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []  # 存储token ID

            # 添加系统提示（若需要）
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            # 处理对话内容
            for conv in source:
                # 兼容不同格式的对话数据
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role = roles.get(role, role)  # 映射角色
                conv = [{"role" : role, "content" : content}]
                conversations.append(content)  # 记录对话内容
                # 应用对话模板，转换为token ID
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            
            # 替换图像和记忆token为模型指定ID
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    
            input_ids.append(input_id)
        # 转换为tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids, conversations  # 返回token ID和对话内容


def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """对齐张量列表（填充到最大长度）"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]  # 隐藏维度
    dtype = tensors[0].dtype
    # 初始化填充后的张量
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)  # 填充值
    # 复制数据到填充后的张量
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output
   
def eval():
    """主函数：解析参数，初始化模型和评估器，执行评估"""
    global local_rank
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="本地进程排名")
    parser.add_argument("--model_path", type=str, default="", help="模型路径")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml', help="Habitat配置文件路径")
    parser.add_argument("--eval_split", type=str, default='val_unseen', help="评估数据集分割")
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln', help="结果输出路径")
    parser.add_argument("--num_future_steps", type=int, default=4, help="未来步骤数")
    parser.add_argument("--num_frames", type=int, default=32, help="每批处理的帧数")
    parser.add_argument("--save_video", default=True, help="是否保存导航视频")
    parser.add_argument("--num_history", type=int, default=8, help="历史帧数")
    parser.add_argument("--model_max_length", type=int, default=4096, help="模型最大序列长度")
    
    # 分布式相关参数
    parser.add_argument('--world_size', default=1, type=int, help='分布式进程数')
    parser.add_argument('--rank', default=0, type=int, help='进程排名')
    parser.add_argument('--gpu', default=0, type=int, help='GPU设备ID')
    parser.add_argument('--port', default='1111', help='分布式通信端口')
    parser.add_argument('--dist_url', default='env://', help='分布式通信URL')
    parser.add_argument('--device', default='cuda', help='设备类型')
    
    args = parser.parse_args()
    # 初始化分布式模式
    init_distributed_mode(args)
    local_rank = args.local_rank

    # 加载分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right"
    )
    
    # 加载模型配置和模型
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation="eager",  # 禁用 FlashAttention2
                # attn_implementation="flash_attention_2",  # 使用FlashAttention加速
                torch_dtype=torch.bfloat16,  # 使用bfloat16精度
                config=config,
                low_cpu_mem_usage=False,
            )
    model.model.num_history = args.num_history  # 设置历史帧数
    model.requires_grad_(False)  # 冻结模型参数
    model.to(local_rank)  # 移动模型到指定设备
    # 执行评估
    evaluate(model, tokenizer, args)


def evaluate(model, tokenizer, args):
    """执行评估：初始化评估器，分布式收集结果并汇总"""
    model.eval()  # 模型设为评估模式
    
    world_size = get_world_size()  # 分布式进程数
    model.reset(world_size)  # 重置模型（分布式适配）
    # 初始化评估器
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args
    )
    # 执行评估，获取当前进程的指标
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 
    
    # 分布式汇总结果
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)  # 收集各进程的episode数量
    # 初始化存储所有进程的指标
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()  # 同步所有进程
    # 收集所有进程的指标
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()  # 同步所有进程
    
    # 合并所有进程的结果
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    # 计算总体指标
    result_all = {
                    "平均成功率": (sum(sucs_all)/len(sucs_all)).item(),
                    "平均SPL": (sum(spls_all)/len(spls_all)).item(),
                    "平均Oracle成功率": (sum(oss_all)/len(oss_all)).item(),
                    "平均到目标距离": (sum(ones_all)/len(ones_all)).item(),
                    "总episode数": len(sucs_all)
                }
    
    print(result_all)  # 打印总体结果
    # 主进程保存总体结果
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()  # 启动评估

import sys
import os
import gzip
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import cv2
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile, ImageDraw, ImageFont
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from transformers.image_utils import to_numpy_array

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

from utils.dist import *
import base64
from datetime import datetime
from io import BytesIO
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
# from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
# from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
from model.qwen.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import AutoProcessor


min_pixels: int = 28 * 28
max_pixels: int = 1605632


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.save_video_ratio = args.save_video_ratio


        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.image_processor = model.processor
        self.model = model
        self.tokenizer = model.tokenizer
        
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "MOVE_FORWARD": [1],
            "TURN_LEFT": [2],
            "TURN_RIGHT": [3]
        })
        self.idx2action = {v[0]: k for k, v in self.actions2idx.items()}
        
        # Load GT data
        # Infer GT path from data_path: "path/to/{split}.json.gz" -> "path/to/{split}_gt.json.gz"
        data_path = self.config.habitat.dataset.data_path.format(split=self.split)
        base, ext = os.path.splitext(data_path)
        if ext == ".gz": # Handle .json.gz
             base, ext2 = os.path.splitext(base)
             ext = ext2 + ext
        gt_path = f"{base}_gt{ext}"
        
        print(f"Loading GT data from {gt_path}")
        with gzip.open(gt_path, 'rt') as f:
            self.gt_data = json.load(f)

        self.num_history = args.num_history


    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env


    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start: ",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                
                if str(episode_id) in self.gt_data:
                    ref_actions = self.gt_data[str(episode_id)]['actions']
                else:
                    ref_actions = []

                env.current_episode = episode
                observations = env.reset()

                vis_frames = []
                step_id = 0
                
                should_save_video = self.save_video and (random.random() < self.save_video_ratio)
                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}'), exist_ok=True)
                

                rgb_list = []
                time_ids = []
                action_seq = []
                self.model.model.past_key_values = None
                
                while not env.episode_over:
                    
                    time_ids.append(step_id)
                    rgb = observations["rgb"]
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    rgb_list.append(image)
                    
                    info = env.get_metrics()
                        
                    history_len = len(rgb_list) - 1 
                    
                    if history_len <= self.num_history:
                        history_images = rgb_list[:history_len]
                        images = history_images + [rgb_list[-1]]
                    else:
                        indices = np.linspace(0, history_len, self.num_history + 1, dtype=int)
                        images = [rgb_list[i] for i in indices]
                    
                    # Modified: Pass episode_id to call_model
                    action = self.model.call_model(images, episode_instruction, step_id, episode_id=episode_id)[0]
                    
                    if info['top_down_map'] is not None and should_save_video:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)

                        frame_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(frame_pil)
                        img_width, img_height = frame_pil.size

                        # --------------------------
                        # 1. Configure text parameters (left layout, larger font size)
                        # --------------------------
                        # Task description
                        task_text = f"Task: {episode_instruction}"
                        # Navigation result information (real-time current metrics)
                        metrics = env.get_metrics()  # Real-time metrics for the current step
                        
                        if isinstance(action, str):
                            current_action_str = action
                        else:
                            current_action_str = self.idx2action.get(action, "UNKNOWN")
                        
                        ref_action_idx = ref_actions[step_id] if step_id < len(ref_actions) else -1
                        ref_action_str = self.idx2action.get(ref_action_idx, "STOP" if ref_action_idx == -1 and step_id >= len(ref_actions) else "UNKNOWN")

                        # Ensure alignment by padding the first column
                        # Calculate padding based on the longest first-column string
                        base_font_size = int(img_height * 0.03) 
                        margin = int(img_height * 0.015) 
                        line_spacing = 4
                        text_color = (255, 255, 255)
                        bg_color = (0, 0, 0, 100)
        
                        font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
                        font = ImageFont.truetype(font_path, base_font_size)
                        # font = ImageFont.load_default()

                        # Helper to measure line height
                        sample_bbox = draw.textbbox((0, 0), "Ay", font=font)
                        single_line_height = sample_bbox[3] - sample_bbox[1] + line_spacing

                        # --------------------------
                        # 2. Text Preparation (Separate Task Wrapping from Stats Grid)
                        # --------------------------
                        max_line_width = int(img_width * 0.41)

                        def wrap_text(text, font, max_width):
                            lines = []
                            current_line = ""
                            for char in text:
                                if char == "\n":
                                    lines.append(current_line)
                                    current_line = ""
                                    continue
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

                        # 1. Wrap Task Description
                        task_lines = wrap_text(f"Task: {episode_instruction}", font, max_line_width)
                        
                        # 2. Episode ID
                        id_line = f"scene_episode ID: {scene_id}_{episode_id}"
                        
                        # 3. Stats Grid Rows (Action + Oracle, Ref + Dist)
                        # We will draw these manually at fixed offsets, not wrap them
                        # Row 1
                        action_lbl = f"action: {current_action_str}"
                        oracle_lbl = f"oracle_success: {metrics['oracle_success']:.1f}"
                        # Row 2
                        ref_lbl = f"ref_action: {ref_action_str}"
                        dist_lbl = f"distance_to_goal: {metrics['distance_to_goal']:.2f}"

                        # Calculate Layout Dimensions
                        # Section 1: Task Lines
                        height_task = len(task_lines) * single_line_height
                        # Section 2: ID Line
                        height_id = single_line_height
                        # Section 3: Stats Rows (2 rows)
                        height_stats = 2 * single_line_height
                        
                        # Extra gap between sections if desired
                        section_gap = 0 # single_line_height * 0.5
                        
                        total_text_height = height_task + section_gap + height_id + section_gap + height_stats

                        # Calculate Max Width for Background
                        # Check Task lines
                        max_w = 0
                        for line in task_lines:
                            bbox = draw.textbbox((0, 0), line, font=font)
                            if (bbox[2] - bbox[0]) > max_w: max_w = bbox[2] - bbox[0]
                        # Check ID Line
                        bbox = draw.textbbox((0, 0), id_line, font=font)
                        if (bbox[2] - bbox[0]) > max_w: max_w = bbox[2] - bbox[0]

                        # Check Stats Width (Action col + Padding + Oracle col)
                        # Determine dynamic col_2_offset based on longest "key" in col 1
                        # The keys are 'action: ...' and 'ref_action: ...'. 
                        # We want the offset to be just enough for the longest likely string + padding.
                        # Usually "ref_action: MOVE_FORWARD" is the longest.
                        # To keep it fixed across frames, we measure "ref_action: MOVE_FORWARD" explicitly, 
                        # or use a safe large fixed string.
                        ref_str_len = draw.textbbox((0, 0), "ref_action: MOVE_FORWARD", font=font)[2]
                        # Add some padding
                        col_2_offset = ref_str_len + int(base_font_size * 3.2)
                        
                        # Check if stats block is wider than max_w
                        # Row 1 width
                        bbox_oracle = draw.textbbox((0, 0), oracle_lbl, font=font)
                        row1_width = col_2_offset + (bbox_oracle[2] - bbox_oracle[0])
                        if row1_width > max_w: max_w = row1_width
                        
                        # Row 2 width
                        bbox_dist = draw.textbbox((0, 0), dist_lbl, font=font)
                        row2_width = col_2_offset + (bbox_dist[2] - bbox_dist[0])
                        if row2_width > max_w: max_w = row2_width

                        # --------------------------
                        # 4. Draw Background Box
                        # --------------------------
                        bg_x1 = margin
                        bg_y1 = margin
                        bg_x2 = bg_x1 + int(max_w) + 2 * margin
                        bg_y2 = bg_y1 + int(total_text_height) + 2 * margin
                        
                        # Clamp
                        if bg_x2 > img_width: bg_x2 = img_width
                        if bg_y2 > img_height: bg_y2 = img_height

                        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)

                        # --------------------------
                        # 5. Draw Text Content
                        # --------------------------
                        curr_y = bg_y1 + margin
                        text_x = bg_x1 + margin

                        # Draw Task
                        for line in task_lines:
                            draw.text((text_x, curr_y), line, font=font, fill=text_color)
                            curr_y += single_line_height
                        
                        # curr_y += section_gap

                        # Draw ID
                        draw.text((text_x, curr_y), id_line, font=font, fill=text_color)
                        curr_y += single_line_height
                        # curr_y += section_gap

                        # Draw Stats Grid
                        # Row 1
                        draw.text((text_x, curr_y), action_lbl, font=font, fill=text_color)
                        draw.text((text_x + col_2_offset, curr_y), oracle_lbl, font=font, fill=text_color)
                        curr_y += single_line_height
                        
                        # Row 2
                        draw.text((text_x, curr_y), ref_lbl, font=font, fill=text_color)
                        draw.text((text_x + col_2_offset, curr_y), dist_lbl, font=font, fill=text_color)
                        
                        # Done Drawing

                        # --------------------------
                        # 6. To Numpy
                        # --------------------------
                        frame = np.array(frame_pil)
                        vis_frames.append(frame)
                    
                    if action in self.actions2idx:
                        action = self.actions2idx[action][0]
                    else:
                        action = 0


                    if step_id >= self.args.max_steps:
                        action = 0

                    observations = env.step(action)
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()
                if should_save_video:
                    # Determine category and sub-category
                    category = ""
                    sub_category = ""
                    
                    if metrics['success'] > 0.001:
                        category = 'success'
                        if step_id >= 350:
                            sub_category = 'long_success'
                        else:
                            sub_category = 'ordinary_success'
                    elif metrics['oracle_success'] > 0.001:
                        category = 'oracle_success'
                        if step_id >= self.args.max_steps:
                            sub_category = 'timeout'
                        else:
                            sub_category = 'stopped_wrong'
                    else: # Fail
                        category = 'fail'
                        if step_id >= self.args.max_steps:
                            sub_category = 'timeout'
                        elif step_id < 5:
                            sub_category = 'stopped_early'
                        else:
                            sub_category = 'stopped_wrong'

                    save_folder = os.path.join(self.output_path, f'vis_{self.epoch}', category, sub_category)
                    os.makedirs(save_folder, exist_ok=True)
                    images_to_video(
                        vis_frames, save_folder, f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                
                if self.args.print_interval > 0 and len(sucs) % self.args.print_interval == 0:
                     print(f"Intermediate Result (Local Rank {get_rank()}) at {len(sucs)} episodes:")
                     current_result = {
                        "sucs": sum(sucs)/len(sucs),
                        "spls": sum(spls)/len(spls),
                        "oss": sum(oss)/len(oss),
                        "ones": sum(ones)/len(ones),
                        'length': len(sucs)
                    }
                     print(current_result)
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

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)     




class VLN_Inference:
    def __init__(self, pretrained, processor_path=None, device="cuda"):
        config = AutoConfig.from_pretrained(pretrained)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained,
            config=config,
            dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            # mode='evaluation'
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        if processor_path is None:
            processor_path = pretrained
        
        self.processor = AutoProcessor.from_pretrained(processor_path, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        
        self.device = device


    def call_model(
        self,
        observations, 
        task,
        step_id,
        add_frame_index: bool=False,
        gen_kwargs: dict = {},
        episode_id=None # Added episode_id parameter
    ):
        
        messages = []
        message = [
                {"role": "system", 
                "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
                }
            ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        for i in enumerate([task]):
    
            visual = observations
            if isinstance(visual, Image.Image): 
                message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
            elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  
                image_content = []
                image_count = 0
                for v in visual:
                    if add_frame_index:
                        image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                    image_content.append({"type": "image", "image": v})
                    image_count += 1
                message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            messages.append(message)

        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
        # Extract images from messages for standard processor
        image_inputs = []
        for message in messages:
            vision_info = extract_vision_info(message)
            for ele in vision_info:
                if "image" in ele:
                    image = ele["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")
                else:
                    raise NotImplementedError("Unsupported vision info type")

                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image_inputs.append(image)

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        )
        device = self.model.device
        inputs = inputs.to(device)

        # ----------------------------------------------------
        # START DEBUG LOGGING
        # ----------------------------------------------------
        if episode_id is not None:
             # Prepare debug data
             debug_data = {
                 "episode_id": episode_id,
                 "step_id": step_id,
                 "instruction": task,
                 "input_ids": inputs.input_ids.cpu().tolist(), # Full list of IDs
             }
             
             # Extract Image Features (Tokens)
             # NOTE: This extracts the precise image token embeddings used by the model
             if "pixel_values" in inputs:
                 try:
                     with torch.no_grad():
                        # The model wrapper has a .model attribute which is the Qwen3VLModel usually, 
                        # but here self.model IS Qwen3VLForConditionalGeneration.
                        # Qwen3VLForConditionalGeneration has get_image_features method.
                         
                         image_grid_thw = inputs.image_grid_thw if "image_grid_thw" in inputs else None
                         video_grid_thw = inputs.video_grid_thw if "video_grid_thw" in inputs else None
                         
                         # Call get_image_features
                         image_embeds, deepstack_image_embeds = self.model.get_image_features(
                             pixel_values=inputs.pixel_values,
                             image_grid_thw=image_grid_thw
                         )
                         
                         # Convert tensor to list (Warning: Large data)
                         # To avoid explosion, we log tuple of shape.
                         # If USER wants full data, uncomment the full conversion.
                         # Based on USER REQUEST "print precise input token", I will include data for the first image
                         # or limit the output. But "print out... to json" implies getting the data.
                         
                         # We'll save all image embeds.
                         # image_embeds is a tuple of tensors (one per image split)
                         debug_data["image_tokens"] = [img_emb.cpu().tolist() for img_emb in image_embeds]
                         debug_data["image_token_shapes"] = [img_emb.shape for img_emb in image_embeds]
                         
                 except Exception as e:
                     debug_data["image_tokens_error"] = str(e)
            
             # Save to JSON
             debug_file = "debug_input_tokens.json"
             # If step_id is 0, maybe clear the file? Or just append?
             # User might run multiple episodes. Append is safer.
             # We should probably put this in an output directory? 
             # For now, current directory or args.output_path if accessible.
             # self.args is not easily accessible here but we can assume CWD or simple file.
             
             with open(debug_file, "a") as f:
                 f.write(json.dumps(debug_data) + "\n")
        # ----------------------------------------------------
        # END DEBUG LOGGING
        # ----------------------------------------------------
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        
        
        pad_token_id = self.tokenizer.pad_token_id
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            # past_key_values=self.model.model.past_key_values,
            # use_cache=True,
        )

        # self.model.model.past_key_values = cont.past_key_values

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("Generated action: ", answers)
        return answers




   
def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument("--save_video_ratio", type=float, default=0.05, help="0~1")
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--max_steps', default=400, type=int,
                        help='max_steps')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    
    parser.add_argument("--print_interval", type=int, default=1000, help="Print evaluation results every N episodes")

    parser.add_argument("--processor_path", type=str, default="/data/home/co/cohw2/Projects/Test/Qwen3/Qwen3-VL-8B-Instruct", help="Path to load processor config from")

    args = parser.parse_args()
    set_seed(args.seed)
    init_distributed_mode(args)
    local_rank = args.local_rank

    model = VLN_Inference(args.model_path, processor_path=args.processor_path, device=f"cuda:{local_rank}")

    evaluate(model, args)



def evaluate(model, args):
    
    world_size = get_world_size()

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()

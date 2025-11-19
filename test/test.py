import os
import torch
from model.qwen.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import AutoProcessor

MODEL = "Qwen3-VL/Qwen3-VL-8B-Instruct"
IMAGE = "018.jpg"
INSTRUCTION = "Walk into the bedroom and stop by the bed." 

# default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     MODEL, dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL)

system_prompt = "You are an autonomous navigation assistant."

user_text = (
    f"Your task is to {INSTRUCTION}. "
    "Based on the current view, output **only the immediate next action** to take. " 
    "Do not output a sequence. "
    "Choose from: TURN LEFT (←), TURN RIGHT (→), MOVE FORWARD (↑), or STOP."
)

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": IMAGE,
            },
            {
                "type": "text", 
                "text": user_text
            },
        ],
    }
]


# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": IMAGE,
#             },
#             {"type": "text", "text": "Describe this image in Chinese."},
#         ],
#     }
# ]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

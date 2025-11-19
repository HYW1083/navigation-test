import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL = "/home/lunet/cohw2/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
IMAGE = "018.jpg"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(MODEL)
messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "image": IMAGE
            },
            {
                "type":"text",
                "text":"Describe this image in Chinese."
            }
        ]
    }

]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs.pop("token_type_ids", None)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

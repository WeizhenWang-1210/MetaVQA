import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image


def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_np(image_array, input_size=448, max_num=12):
    """
    Assuming image_array is already in RGB channeling
    """
    image = Image.fromarray(image_array).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_model(model_path):
    """
    Our implementation for CVPR2025 based on Transformers. You can also use other libraries.
    """
    if "qwen2" in model_path.lower():
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
    elif "llama" in model_path.lower():
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
        ).eval()
    elif "internvl" in model_path.lower():
        return load_internvl(model_path)
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, processor, tokenizer

from transformers import AutoConfig
import json
from pathlib import Path
import os
from safetensors.torch import load_file 


def load_internvl(model_path):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True
    ).eval()
    processor = None #AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, processor, tokenizer



def inference(model, processor, tokenizer, prompt, obs):
    """
    Our implementation for CVPR2025 based on Transformers. You can also use other libraries.
    """
    def prepare_prompt(prompt):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                    "text": f"{prompt}"}
                ],
            }
        ]
        return conversation    
    conversation = prepare_prompt(prompt=prompt)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    #obs in RGB
    raw_image = Image.fromarray(obs)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    response = processor.decode(output[0], skip_special_tokens=True)
    return response 


def inference_internvl(model, processor, tokenizer, prompt, obs):
    """
    Our implementation for CVPR2025. Assuming obs(np.array) is already in RGB channeling
    """
    pixel_values = load_image_np(obs, max_num=12).to(torch.bfloat16).to("cuda")
    generation_config = dict(max_new_tokens=512, do_sample=False)
    question = f'<image>\n{prompt}'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


def inference_internvl_zeroshot(model, processor, tokenizer, prompt, obs):
    """
    Our implementation for CVPR2025
    """
    pixel_values = load_image_np(obs, max_num=12).to(torch.bfloat16).to("cuda")
    generation_config = dict(max_new_tokens=512, do_sample=False)
    question = f'<image>\n{prompt}'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    valid_choices = ["(A)", "(B)", "(C)", "(D)", "(E)", "(a)", "(b)", "(c)", "(d)", "(e)"]
    for valid_choice in valid_choices:
        if valid_choice in response:
            return valid_choice[1]
    return ""


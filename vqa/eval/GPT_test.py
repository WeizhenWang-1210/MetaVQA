import asyncio
import os.path

import aiohttp
import json
from openai import OpenAI, AsyncOpenAI
from collections import defaultdict
import base64

API_KEY = None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def respond_vqa(session, base64_image, text, api_key, index):
    print(f"Answering {index}th data.")
    url = "https://api.openai.com/v1/chat/completions"  # Adjust if API endpoint changes
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o",  # Adjust based on available models and your preference
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Please answer the questions as provided."},
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"{text}\nPlease respond in the following format:\n{{\n\t\"REASONING\":\"<Your Reasoning>\"\n\t\"ANSWER\":\"<Your Answer>\"\n}}\n"
                },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
    }
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            data = await response.json()
            # assert "choices" in data.keys(), f"{index}th response don't have choice,why?"
            # print(data["choices"][0]['message']['content'])
            # print(data)
            return data["choices"][0]['message']['content']  # Adjust based on the response structure
    except Exception as e:
        print(f"Request failed {index}: {e}")


async def get_responses(results, vqas, api_key):
    async with aiohttp.ClientSession() as session:
        # Prepare tasks only for rows where the new_column is empty
        tasks = []
        indices = []  # To keep track of row indices for assigning responses correctly
        for index, vqa in vqas.items():
            if results[index] == "":
                print(f"Send {index}th question to be processed")
                im = encode_image(vqa["obs"][0])
                tasks.append(respond_vqa(session, im, vqa["question"], api_key, index))
                indices.append(index)
        # Execute all tasks asynchronously
        responses = await asyncio.gather(*tasks)
        # Assign responses to the DataFrame
        for index, response in zip(indices, responses):
            results[index] = response


async def fill_qa(results, vqas, api_key):
    async with aiohttp.ClientSession() as session:
        # Prepare tasks only for rows where the new_column is empty
        tasks = []
        indices = []  # To keep track of row indices for assigning responses correctly
        for index, vqa in vqas.items():
            if "model_response" not in results[index].keys() or results[index]["model_response"] == None or results[index]["model_response"] == "":
                print(f"Send {index}th question to be processed")
                im = encode_image(vqa["obs"][0])
                tasks.append(respond_vqa(session, im, vqa["question"], api_key, index))
                indices.append(index)
        # Execute all tasks asynchronously
        responses = await asyncio.gather(*tasks)
        # Assign responses to the DataFrame
        for index, response in zip(indices, responses):
            results[index]["model_response"] = response


import tqdm

if __name__ == "__main__":
    """
    Inference script for benchmarking GPT-4o on MetaVQA-Train.
    """
    qa_path = "/data_weizhen/metavqa_cvpr/datasets/test/test/test_processed.json"
    save_path = "/experiments/main/GPT4o_test_results.json"
    data_all = json.load(open(qa_path, "r"))
    if os.path.exists(save_path):
        data_saved = json.load(open(save_path, "r"))
    else:
        import copy
        data_saved = copy.deepcopy(data_all)
    batchsize = 10
    for _ in range(10):
        for start in tqdm.tqdm(range(0, len(list(data_all.keys())), batchsize), desc="Sending jobs", unit="batch"):
            data = dict(list(data_all.items())[start:start + batchsize])
            asyncio.run(fill_qa(data_saved, data, API_KEY))
    json.dump(
        data_saved, open(save_path, "w"), indent=2
    )

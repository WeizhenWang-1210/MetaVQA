import asyncio
import aiohttp
import json
from openai import OpenAI, AsyncOpenAI
from collections import defaultdict
import base64

API_KEY = None

PROMPT = "Can you propose some question-answer pairs using the labeling present in the image? These pairs are used to convert general-purpose VLMs into embodied agents in driving scenarios. " \
         "I want two types of questions: (1) \"spatial\" questions, designed to improve the spatial reasoning capability of VLMs. (2)\"embodied\" questions, designed for VLMs to counterfactually" \
         "examine the consequences of possible actions. Please organize your generated response in a JSON-compatible Python string with four fields: (1) \"question,\" the question you formulated " \
         "in a multiple-choice setting, including the option strings. (2) \"answer,\" a single capitalized character indicating the chosen option. (3) \"explanation,\" the reasoning in order to " \
         "reach the \"answer.\" (4)\"type\", whether the question is \"spatial\" or \"embodied\". Please refer to objects using their labels. For example, if a car is bounded by a box with the label" \
         "\"1\", refer to the car by \"<1>\". In addition, you can leave out the number ordering for each question. Put the question-answer pairs in python parsable json strings."


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def respond_vqa(session, base64_image, api_key, index):
    print(f"Annotating {index}th image.")
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
                    "text": f"{PROMPT}"
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
            #assert "choices" in data.keys(), f"{index}th response don't have choice,why?"
            #print(data["choices"][0]['message']['content'])
            #print(data)
            return data["choices"][0]['message']['content']  # Adjust based on the response structure
    except Exception as e:
        print(f"Request failed {index}: {e}")


async def fill_qa(results, vqas, api_key):
    async with aiohttp.ClientSession() as session:
        # Prepare tasks only for rows where the new_column is empty
        tasks = []
        indices = []  # To keep track of row indices for assigning responses correctly
        for index, vqa in vqas.items():
            if index not in results.keys() or "model_response" not in results[index].keys() or results[index][
                "model_response"] is None or results[index]["model_response"] == "":
                print(f"Send {index}th question to be processed")
                im = encode_image(vqa["obs"][0])
                tasks.append(respond_vqa(session, im, api_key, index))
                indices.append(index)
        # Execute all tasks asynchronously
        responses = await asyncio.gather(*tasks)
        # Assign responses to the DataFrame
        for index, response in zip(indices, responses):
            if index not in results.keys():
                results[index] = dict(model_response="", obs=[])
            results[index]["model_response"] = response
            results[index]["obs"] = vqas[index]["obs"]
            results[index]["domain"] = vqas[index]["domain"]
            results[index]["world"] = vqas[index]["world"]


import time
import tqdm

if __name__ == "__main__":
    qa_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/rebuttal_processed.json"
    save_path = "/data_weizhen/metavqa_cvpr/datasets/trainval/experiments/gpt_annotated.json"
    data_all = json.load(open(qa_path, "r"))
    try:
        data_saved = json.load(open(save_path, "r"))
    except:
        print("No gpt_annotated.json exists. Creating an empty dictionary")
        data_saved = {}
    #data_all = dict(list(data_all.items())[:10])
    print(f"{len(data_saved)} already generated.")
    batchsize = 5
    for _ in range(10):
        #print(f"{len(results)} out of {len(data_all)} responded.")
        for start in tqdm.tqdm(range(0, len(list(data_all.keys())), batchsize), desc="Sending jobs", unit="batch"):
            if start + batchsize < len(data_all):
                data = dict(list(data_all.items())[start:start + batchsize])
            else:
                data = dict(list(data_all.items())[start:])
            asyncio.run(fill_qa(data_saved, data, API_KEY))
    json.dump(
        data_saved, open(save_path, "w"), indent=2
    )

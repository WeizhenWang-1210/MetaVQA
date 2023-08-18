import objaverse
import json
import pprint
from tqdm import tqdm
import openai
import multiprocessing
import asyncio
import trimesh
#car packs 20f9af9b8a404d5cb022ac6fe87f21f5
#landrover ffba8330b24d42daac8905fa0102eb97
#2021 Lamborghini Countach LPI 800-4 d76b94884432422b966d1a7f8815afb5
#Porsche 911 Carrera 4S d01b254483794de3819786d93e0e1ebf
#pickup 40c94d8b31f94df3bd80348cac8624f1
#Mahindra thar 4by4 f045413d71d743c58682881cb7421d64
class Objverse_helper:
    def __init__(
            self,
            target_prompt = "Please filter the list to only keep words that are strictly a wellknowned type of vehicle and have no other meaning.",
            openai_key = "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"
        ):
        self.target_prompt = target_prompt
        openai.api_key =  openai_key
    def getAllTag(self, num_uids=-1):
        uids = objaverse.load_uids()
        print("Load all annotations...")
        annotations = objaverse.load_annotations(uids[:num_uids])
        tag = set()
        for uid, content in tqdm(annotations.items(), desc="Parsing all annotations"):
            for eachtag in content["tags"]:
                tag.add(eachtag["name"])
        return annotations, tag
    def getTagStrictly(self, num_uids=-1, target_tag = "car"):
        uids = objaverse.load_uids()
        print("Load all annotations...")
        annotations = objaverse.load_annotations(uids[:num_uids])
        full_tag = []
        uid_list = []
        full_list = []
        for uid, content in tqdm(annotations.items(), desc="Parsing all annotations"):
            for eachtag in content["tags"]:
                full_tag.append(eachtag['name'])
                if eachtag['name'] == target_tag:
                    uid_list.append(uid)
                    full_list.append(content)
        return uid_list, full_list, full_tag
    def filterTagGPT(self, taglist):
        query_list = [
            "I will give you a list of words.",
            self.target_prompt,
            "Here is the list: {}".format(','.join(taglist)),
            "Please directly return the filtered list without anything else"
        ]
        messages = [
            {"role": "system",
             "content": "You are a expert on classifying things and will select specific words that match specific criteria from a wordlist."},
            {"role": "user", "content": " ".join(query_list)},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
        )
        # print(response)
        response_message = response["choices"][0]["message"]["content"]
        return response_message
    def parseTagGPT(self, total_taglist, num_per_query = 20):
        for i in range(0, len(total_taglist)-num_per_query, num_per_query):
            print(self.filterTagGPT(total_taglist[i:i+num_per_query]))

if __name__ == "__main__":
    processes = multiprocessing.cpu_count()
    objhelper = Objverse_helper()
    uid_list, full_list, full_tag = objhelper.getTagStrictly(-1, "car")
    objects = objaverse.load_objects(
        uids=uid_list,
        download_processes=processes
    )
    with open("/home/chenda/objpath.txt", "w") as f:
        for uid, path in objects.items():
            f.write("{}:{}".format(uid,path))

    #uids = objaverse.load_uids()
    # objects = [
    #     "ffba8330b24d42daac8905fa0102eb97",
    # ]
    # objects = objaverse.load_objects(objects,1)
    # trimesh.load(list(objects.values())[0]).show()
    # #trimesh.load("metadrive/assets/models/lambo/vehicle.glb").show()
    #
    # car_spec = {
    #     "lambo":{
    #         'length':4.87,
    #         'width':2.099,
    #         'height':1.139,
    #         'mass':1595,
    #         'wheelbase':2.7,
    #         'max_speed':355,
    #         'front_wheel_width':0.255,
    #         'back_wheel_width':0.355
    #         #'front_wheel_diameter': 2 * 25.5 * 0.3 + 20*2.54 15.3 + 50.8
    #         #'back_wheel_diameter':  2 * 25.5 * 0.25 + 21*2.54
    #     }
    # }


    """
    LAMBO Spec
    TIRE_RADIUS = 0.3305#0.313
    TIRE_WIDTH = 0.255#0.25
    MASS = 1595#1100
    LATERAL_TIRE_TO_CENTER = 1#0.815
    FRONT_WHEELBASE = 1.36#1.05234
    REAR_WHEELBASE = 1.45#1.4166
    #path = ['ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]
    path = ['lambo/vehicle.glb', (0.5,0.5,0.5), (1.09, 0, 0.6), (0, 0, 0)]"""

   
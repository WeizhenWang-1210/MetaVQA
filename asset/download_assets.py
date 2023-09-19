import objaverse
import json
import pprint
import os
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
    def getTagStrictly(self, num_uids=-1, target_tag = "car", strict=True):
        uids = objaverse.load_uids()
        print("Load all annotations...")
        annotations = objaverse.load_annotations(uids[:num_uids])
        full_tag = []
        uid_list = []
        full_list = []
        for uid, content in tqdm(annotations.items(), desc="Parsing all annotations"):
            for eachtag in content["tags"]:
                full_tag.append(eachtag['name'])
                if strict:
                    if eachtag['name'] == target_tag:
                        uid_list.append(uid)
                        full_list.append(content)
                else:
                    if target_tag in eachtag["name"]:
                        uid_list.append(uid)
                        full_list.append(content)
        return uid_list, full_list, full_tag
    def getTagList(self, num_uids=-1, target_tag_list = []):
        uids = objaverse.load_uids()
        print("Load all annotations...")
        annotations = objaverse.load_annotations(uids[:num_uids])
        full_tag = []
        uid_list = []
        full_list = []
        for uid, content in tqdm(annotations.items(), desc="Parsing all annotations"):
            for eachtag in content["tags"]:
                full_tag.append(eachtag['name'])
                if eachtag["name"] in target_tag_list:
                    uid_list.append(uid)
                    full_list.append(content)
        return uid_list, full_list, full_tag
    def saveTag(self, objects, tagname, parent_folder, isTagList = False, tagList = None):
        if isTagList:
            assert tagList is not None
        if not isTagList:
            path = "object-paths-{}.json".format(tagname)
        else:
            path = "object-list-paths-{}.json".format(tagname)
        with open(os.path.join(parent_folder, path), "w") as f:
            if isTagList:
                objects["tagList"] = tagList
            json.dump(objects, f)


if __name__ == "__main__":
    processes = 16
    objhelper = Objverse_helper()
    # _, tag = objhelper.getAllTag()
    # with open("C:\\research\\dataset\\alltag.txt", "w+") as f:
    #     for each in tag:
    #         f.write("{}\n".format(each))
    # tag = "pedestrian"
    # uid_list, full_list, full_tag = objhelper.getTagStrictly(-1, tag, strict=False)
    # print(len(uid_list))
    # objects = objaverse.load_objects(
    #     uids=uid_list[:100],
    #     download_processes=processes
    # )
    #
    # path =  "object-paths-{}.json".format(tag)
    # objhelper.saveTag(objects, tagname=tag, parent_folder="C:\\research\\dataset\\.objaverse\\hf-objaverse-v1",
    #                   isTagList=False)
    tag = "crosswalk"
    taglist = ["crosswalksign", "americancrosswalksign", "crosswalk", "americancrosswalk", ]


    uid_list, full_list, full_tag = objhelper.getTagList(num_uids=-1, target_tag_list = taglist)
    uid_list.sort()
    print(len(uid_list))
    objects = objaverse.load_objects(
        uids=uid_list[:150],
        download_processes=processes
    )
    objhelper.saveTag(objects,tagname=tag, parent_folder="C:\\research\\dataset\\.objaverse\\hf-objaverse-v1", isTagList=True, tagList=taglist)

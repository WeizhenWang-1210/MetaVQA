import json
import os
import shutil
import tqdm
'''
-root
    -episodes
        -episode_1
            -frame0
                -leftb, rightb, leftf, rightf, f, b
            -frame1
                -....
        -episode_2
            -.....
    -qas
        -train
            -json1
            -json2
            -json3
            ...
        -val
            -json1
            -json2
            -json3
            ....
        -test
            -json1
            -json2
            -json3
            ....
    index_file # know all episodes, vqa(with corresponding spit.)
    
    
    
    
    index_file: aggregated qa id
    -ids
        -global id: {file_path:'...', local_id:'...'}
    -splits
        -train
            [global id 1, global id 2....]
        -val
            [global id 1, global id 2, ....]
        test
            [global id 1, global id 2, ....]
            
'''


def copy_files(src_folder, dest_folder):
    #print(src_folder, dest_folder)
    os.makedirs(dest_folder)
    for item in os.listdir(src_folder):
        source_path = os.path.join(src_folder, item)
        destination_path = os.path.join(dest_folder, item)

        if os.path.isdir(source_path):
            # Copy directory
            shutil.copytree(source_path, destination_path)
        else:
            # Copy file
            shutil.copy2(source_path, destination_path)


def aggregate_qa(qa_jsons, split_info, root_folder, version, timestamp, map = None):
    '''
    :param qa_jsons: list of qa_json paths
    :return: a dict with all qa
    '''
    index = dict(
        data=dict(),
        split=dict(
            train=[],
            val=[],
            test=[]),
        version=version,
        timestamp=timestamp
    )
    global_idx = 0
    emap = dict() if map is None else map
    episode_idx = 0 if map is None else len(map.keys())
    for qa_json in tqdm.tqdm(qa_jsons,desc="Files", position=0):
        qa = json.load(open(qa_json, 'r'))
        split = split_info[qa_json]
        for local_id, qa_tuple in tqdm.tqdm(qa.items(), desc=f"Progress for {qa_json}",position=1):
            #copy observations by establishing new episodes
            #print(qa_tuple["rgb"]["front"][0])
            old_episode_path = "/".join(qa_tuple["rgb"]["front"][0].split("/")[:-2])
            if old_episode_path in emap.keys():
                now_episode_path = emap[old_episode_path]
            else:
                now_episode_path = os.path.join("episodes", f"episode_{episode_idx}")
                absolute_now_episode_path = os.path.join(root_folder, "episodes", f"episode_{episode_idx}")
                #print(old_episode_path)
                #return
                copy_files(old_episode_path, absolute_now_episode_path)
                #return
                emap[old_episode_path] = now_episode_path
                episode_idx += 1
            new_qa_tuple = dict(
                question=qa_tuple["question"], answer=qa_tuple["answer"], type=qa_tuple["question_type"],
                rgb=dict(
                    front=[], leftb=[], leftf=[], rightb=[], rightf=[], back=[]
                )
            )
            for perspective, imgs in qa_tuple["rgb"].items():
                for img in imgs:
                    new_qa_tuple["rgb"][perspective].append(
                        img.replace(old_episode_path, now_episode_path)
                    )
            index["data"][global_idx] = new_qa_tuple
            index["split"][split].append(global_idx)
            global_idx += 1
    return index, emap


from datetime import datetime




def main():
    if "mapping.json" in os.listdir("/bigdata/weizhen/metavqa_demo/vqa_packed"):
        print("Found mapping.json. Will use it for file copying.")
        established_mapping = json.load(open(os.path.join("/bigdata/weizhen/metavqa_demo/vqa_packed","mapping.json")))
    else:
        print("No mapping.json exist.")
        established_mapping = None
    version = "0.1"
    timestamp = str(datetime.now())
    jsons = [
        "dynamic_qa20.json", "static_qa16.json"#, "static_qa9.json"
    ]
    jsons = [os.path.join("/bigdata/weizhen/metavqa_demo/vqa_raw", name) for name in jsons]
    split = {
        name: "train" for name in jsons
    }
    merged, mapping = aggregate_qa(jsons, split, "/bigdata/weizhen/metavqa_demo/vqa_packed", version, timestamp, established_mapping)
    dest, map = os.path.join("/bigdata/weizhen/metavqa_demo/vqa_packed", "data.json"), os.path.join("/bigdata/weizhen/metavqa_demo/vqa_packed", "mapping.json")
    json.dump(
        merged, open(dest,"w"), indent=2
    )
    json.dump(
        mapping, open(map,"w"), indent=2
    )
def aggregate__real_qa(qa_jsons, split_info, root_folder, version, timestamp, map = None):
    '''
    :param qa_jsons: list of qa_json paths
    :return: a dict with all qa
    '''
    index = dict(
        data=dict(),
        split=dict(
            train=[],
            val=[],
            test=[]),
        version=version,
        timestamp=timestamp
    )
    global_idx = 0
    emap = dict() if map is None else map
    episode_idx = 0 if map is None else len(map.keys())
    for qa_json in tqdm.tqdm(qa_jsons,desc="Files", position=0):
        qa = json.load(open(qa_json, 'r'))
        split = split_info[qa_json]
        for local_id, qa_tuple in tqdm.tqdm(qa.items(), desc=f"Progress for {qa_json}",position=1):
            #copy observations by establishing new episodes
            #print(qa_tuple["rgb"]["front"][0])
            old_episode_path = "/".join(qa_tuple["real"]["front"][0].split("/")[:-2])
            if old_episode_path in emap.keys():
                now_episode_path = emap[old_episode_path]
            else:
                now_episode_path = os.path.join("episodes", f"episode_{episode_idx}")
                absolute_now_episode_path = os.path.join(root_folder, "episodes", f"episode_{episode_idx}")
                #print(old_episode_path)
                #return
                copy_files(old_episode_path, absolute_now_episode_path)
                #return
                emap[old_episode_path] = now_episode_path
                episode_idx += 1
            new_qa_tuple = dict(
                question=qa_tuple["question"], answer=qa_tuple["answer"], type=qa_tuple["question_type"],
                rgb=dict(
                    front=[], leftb=[], leftf=[], rightb=[], rightf=[], back=[]
                )
            )
            for perspective, imgs in qa_tuple["real"].items():
                for img in imgs:
                    new_qa_tuple["rgb"][perspective].append(
                        img.replace(old_episode_path, now_episode_path)
                    )
            index["data"][global_idx] = new_qa_tuple
            index["split"][split].append(global_idx)
            global_idx += 1
    return index, emap
def main_real():
    if "mapping.json" in os.listdir("/bigdata/weizhen/metavqa_demo/vqa_packed"):
        print("Found mapping.json. Will use it for file copying.")
        established_mapping = json.load(open(os.path.join("/bigdata/weizhen/metavqa_demo/vqa_packed","mapping.json")))
    else:
        print("No mapping.json exist.")
        established_mapping = None
    version = "0.1"
    timestamp = str(datetime.now())
    jsons = os.listdir("/bigdata/weizhen/metavqa_demo/vqa_processed")
    jsons = [os.path.join("/bigdata/weizhen/metavqa_demo/vqa_processed", name) for name in jsons if "nuscene" in name]
    split = {
        name: "train" for name in jsons
    }
    merged, mapping = aggregate_qa(jsons, split, "/bigdata/weizhen/metavqa_demo/vqa_packed", version, timestamp, established_mapping)
    dest, map = os.path.join("/bigdata/weizhen/metavqa_demo/vqa_packed", "data_real.json"), os.path.join("/bigdata/weizhen/metavqa_demo/vqa_packed", "mapping.json")
    json.dump(
        merged, open(dest,"w"), indent=2
    )
    json.dump(
        mapping, open(map,"w"), indent=2
    )


if __name__ == "__main__":
    #main()
    qas = json.load(open("/bigdata/weizhen/metavqa_demo/vqa_packed/data.json","r"))
    print(len(qas["data"]))
    print(qas["data"]["0"])
    #print(qas["split"])
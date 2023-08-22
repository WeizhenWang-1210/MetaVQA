import objaverse
import os
import trimesh
import multiprocessing


class Objverse_filter_asset:
    def __init__(self, asset_folder_path="F:\\metaasset\\hf-objaverse-v1"):
        self.asset_folder_path = asset_folder_path
        self.json_gz_path = os.path.join(asset_folder_path, "object-paths.json.gz")
        self.json_path = os.path.join(asset_folder_path, "object-paths.json")
        self.cached_asset_uids = []

    def load_cached_uids(self):
        uids = []

        # Traverse the directory
        for dirpath, dirnames, filenames in os.walk(self.asset_folder_path):
            # Check if we are inside a "glbs" subfolder and the pattern "xxx-xxx" exists in the path
            if "glbs" in dirpath and any('-' in d and d.split('-')[0].isdigit() and d.split('-')[1].isdigit() for d in
                                         dirpath.split(os.sep)):
                for filename in filenames:
                    # Check if the file has a '.glb' extension
                    if filename.endswith('.glb'):
                        # Extract uid from the filename (assuming uid is the filename without the extension)
                        uid = filename.replace('.glb', '')
                        uids.append(uid)

        print(len(uids))
        return uids

    def show_mesh_in_process(self, mesh_path, queue):
        mesh = trimesh.load(mesh_path)
        mesh.show()
        queue.put("done")

    def filter_uid_raw(self, uids):
        objects = objaverse.load_objects(uids=uids)
        queue = multiprocessing.Queue()

        for mesh_path in objects.values():
            p = multiprocessing.Process(target=self.show_mesh_in_process, args=(mesh_path, queue))
            p.start()

            while True:
                if not queue.empty():
                    if queue.get() == "done":
                        break

                user_input = input("Close current window and go to the next mesh? (y/n): ")

                if user_input == "y":
                    p.terminate()
                    p.join()
                    break
                elif user_input == "n":
                    p.terminate()
                    p.join()
                    return

if __name__ == "__main__":
    objaverse_filter_helper = Objverse_filter_asset()
    cached_uid_lists = objaverse_filter_helper.load_cached_uids()
    objaverse_filter_helper.filter_uid_raw(cached_uid_lists)
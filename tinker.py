import objaverse
import asyncio
import trimesh
if __name__ == "__main__":
    #uids = objaverse.load_uids()
    objects = objaverse.load_objects(["d76b94884432422b966d1a7f8815afb5"],1)
    trimesh.load(list(objects.values())[0]).show()
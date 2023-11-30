import objaverse
import json
import pprint
from tqdm import tqdm
import multiprocessing
import asyncio
import trimesh

objects = objaverse.load_objects(uids=['6c3a32958c2d43cdbf12a7109616bdbe'])
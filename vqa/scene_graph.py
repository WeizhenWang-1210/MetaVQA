import math
from typing import Iterable
from collections import defaultdict
from vqa.object_node import ObjectNode, TemporalNode
import json
from vqa.dataset_utils import transform_heading
from vqa.object_node import transform_vec


class SceneGraph:
    def __init__(self,
                 ego_id: str,
                 nodes: list = [],
                 folder: str = "./"
                 ):
        self.nodes: dict[str, ObjectNode] = {}
        for node in nodes:
            self.nodes[node.id] = node
        self.ego_id: str = ego_id
        self.spatial_graph: dict = self.compute_spatial_graph()
        self.folder = folder
        self.statistics = self.generate_statistics()

    def refocus(self, new_ego_id: str) -> None:
        """
        Change to ego_id to the new ego_id
        """
        self.ego_id = new_ego_id

    def compute_spatial_graph(self) -> dict:
        def compute_spatial_edges(ego_id: str, ref_id: str) -> dict:
            edges = {
                'l': [],
                'r': [],
                'f': [],
                'b': [],
                'lf': [],
                'rf': [],
                'lb': [],
                'rb': [],
                's': []
            }
            ego_node = self.nodes[ego_id]
            ref_node = self.nodes[ref_id]
            for node_id, node in self.nodes.items():
                if node_id != ego_id:
                    relation = ego_node.compute_relation(node, ref_node.heading)
                    side, front = relation['side'], relation['front']
                    if side == -1 and front == 0:
                        edges['l'].append(node.id)
                    elif side == -1 and front == -1:
                        edges['lb'].append(node.id)
                    elif side == -1 and front == 1:
                        edges['lf'].append(node.id)
                    elif side == 0 and front == -1:
                        edges['b'].append(node.id)
                    elif side == 0 and front == 1:
                        edges['f'].append(node.id)
                    elif side == 1 and front == 0:
                        edges['r'].append(node.id)
                    elif side == 1 and front == -1:
                        edges['rb'].append(node.id)
                    elif side == 1 and front == 1:
                        edges['rf'].append(node.id)
                    else:
                        # print("Erroenous Relations!\n{}:{},\n{}:{}".format(ego_node.id, ego_node.pos, node.id, node.pos))
                        # exit()
                        edges['s'].append(node.id)
            return edges

        graph = {}
        for node_id in self.nodes.keys():
            graph[node_id] = compute_spatial_edges(node_id, self.ego_id)
        return graph

    def get_nodes(self) -> Iterable[ObjectNode]:
        return list(self.nodes.values())

    def get_ego_node(self) -> ObjectNode:
        return self.nodes[self.ego_id]

    def get_node(self, id: str) -> ObjectNode:
        return self.nodes[id]

    def generate_statistics(self) -> dict:
        colors, types = set(), set()
        for node in self.nodes.values():
            if node.visible:
                colors.add(node.color)
                types.add(node.type)
        return {
            "<p>": list(colors),
            "<t>": list(types)
        }


class TemporalGraph:
    def __init__(self, framepaths, observable_at_key=True, tolerance=0.8):
        """
        We ask questions based on the observation_phase and return answr for the prediction phase
        Note that in MetaDrive each step is 0.1s
        Will give attempt to give 2 seconds of observation and half seconds of prediction phase
        So, observation phase = 20 and prediction phase = 5
        Each graph will store the path to the original annotation("For statistics purpose") and also the loaded information

        """

        self.observation_phase = 0.8  # This is the percentage of frames belonging into observation. The last frame
        # is "present"
        self.prediction_phase = 1.0 - self.observation_phase
        self.tolerance = tolerance  # The percentage(of observation phase) of being observable for objects to be
        # considered in the graph.
        self.observable_at_key = observable_at_key  # enforce that the objects referred must be observable at the key
        # frame.
        self.framepaths: list[str] = framepaths
        self.frames: list[dict] = [json.load(open(framepath, "r")) for framepath in framepaths]

        self.num_observation_frames = math.floor(len(self.framepaths) * self.observation_phase)
        self.num_prediction_frames = len(self.framepaths) - self.num_observation_frames
        self.idx_key_frame = self.num_observation_frames - 1  # since 0-indexed. The past is [0,self._idx_key_frame]
        self.node_ids: Iterable[str] = self.find_node_ids(self.frames, self.observable_at_key, self.tolerance)
        self.ego_id: str = self.frames[0]["ego"]["id"]
        assert all([self.frames[i]["ego"]["id"] == self.ego_id for i in
                    range(len(self.frames))]), "Ego changed during this period."
        self.nodes: dict[str, TemporalNode] = self.build_nodes(self.node_ids, self.frames)
        for i in self.nodes.keys():
            for j in self.nodes.keys():
                if j != i:
                    self.nodes[i].analyze_interaction(self.nodes[j], self.idx_key_frame)
        self.key_frame_graph: SceneGraph = self.build_key_frame()
        self.statistics = self.generate_statistics()

    def get_ego_node(self):
        return self.nodes[self.ego_id]

    def get_node(self, node_id):
        return self.nodes[node_id]

    def build_nodes(self, node_ids, frames) -> dict[str, TemporalNode]:
        # print(len(frames))
        positions = defaultdict(list)
        headings = defaultdict(list)
        colors = defaultdict(str)
        types = defaultdict(str)
        observing_cameras = defaultdict(list)
        speeds = defaultdict(list)
        bboxes = defaultdict(list)
        heights = defaultdict(list)
        states = defaultdict(list)
        collisions = defaultdict(list)
        temporal_nodes = {}
        for timestamp, frame in enumerate(frames):
            for object in frame["objects"]:
                if object["id"] in node_ids:
                    positions[object["id"]].append(object["pos"])
                    headings[object["id"]].append(object["heading"])
                    colors[object["id"]] = object["color"]
                    types[object["id"]] = object["type"]
                    observing_cameras[object["id"]].append(object["observing_camera"])
                    speeds[object["id"]].append(object["speed"])
                    bboxes[object["id"]].append(object["bbox"])
                    heights[object["id"]].append(object["height"])
                    states[object["id"]].append(object["states"])
                    collisions[object["id"]] += [(timestamp, id) for (_, id) in object["collisions"]]
            positions[self.ego_id].append(frame["ego"]["pos"])
            headings[self.ego_id].append(frame["ego"]["heading"])
            colors[self.ego_id] = frame["ego"]["color"]
            types[self.ego_id] = frame["ego"]["type"]
            observing_cameras[self.ego_id].append(frame["ego"]["observing_camera"])
            speeds[self.ego_id].append(frame["ego"]["speed"])
            bboxes[self.ego_id].append(frame["ego"]["bbox"])
            heights[self.ego_id].append(frame["ego"]["height"])
            states[self.ego_id].append(frame["ego"]["states"])
            collisions[self.ego_id] += [(timestamp, id) for (_, id) in frame["ego"]["collisions"]]
        for node_id in node_ids:
            temporal_node = TemporalNode(
                id=node_id, now_frame=self.idx_key_frame, type=types[node_id], height=heights[node_id],
                positions=positions[node_id],
                color=colors[node_id], speeds=speeds[node_id], headings=headings[node_id], bboxes=bboxes[node_id],
                observing_cameras=observing_cameras[node_id], states=states[node_id], collisions=collisions[node_id],
            )
            temporal_nodes[node_id] = temporal_node
        return temporal_nodes

    def find_node_ids(self, frames, observable_at_key=True, noise_tolerance=0.8) -> Iterable[str]:
        """
        Not including the ego_id
        We consider objects that are observable for the noise_tolerance amount
        of time(also, must be observable at key frame) for the observation period and still exist in the prediction phase
        """
        observation_frames = self.num_observation_frames
        prediction_frames = self.num_prediction_frames
        observable_at_t = {
            t: set() for t in range(observation_frames)
        }
        exist_at_t = {
            t: set() for t in range(observation_frames + prediction_frames)
        }
        all_nodes = set()
        for i in range(observation_frames):
            for obj in frames[i]["objects"]:
                if obj["visible"]:
                    observable_at_t[i].add(obj["id"])
                all_nodes.add(obj["id"])
                exist_at_t[i].add(obj["id"])
        for i in range(observation_frames, observation_frames + prediction_frames):
            for obj in frames[i]["objects"]:
                exist_at_t[i].add(obj["id"])
                all_nodes.add(obj["id"])
        final_nodes = set()
        for node in all_nodes:
            observable_frame = 0
            for nodes_observable in observable_at_t.values():
                if node in nodes_observable:
                    observable_frame += 1
            if observable_frame / observation_frames >= noise_tolerance:
                if (not observable_at_key) or (node in observable_at_t[self.idx_key_frame]):
                    final_nodes.add(node)
        for nodes_exist in exist_at_t.values():
            final_nodes = final_nodes.intersection(nodes_exist)
        final_nodes.add(frames[0]["ego"]["id"])
        return final_nodes

    def get_nodes(self):
        return self.nodes

    def build_key_frame(self) -> SceneGraph:
        key_frame_annotation = self.frames[self.idx_key_frame]  # json.load(open(key_frame_path, "r"))
        nodes = []
        ego_id = key_frame_annotation["ego"]["id"]
        ego_dict = key_frame_annotation['ego']
        ego_node = ObjectNode(
            pos=ego_dict["pos"],
            color=ego_dict["color"],
            speed=ego_dict["speed"],
            heading=ego_dict["heading"],
            id=ego_dict['id'],
            bbox=ego_dict['bbox'],
            height=ego_dict['height'],
            type=ego_dict['type'],
            lane=ego_dict['lane'],
            visible=ego_dict["visible"],
            states=ego_dict["states"],
            collisions=ego_dict["collisions"]
        )
        nodes.append(ego_node)
        for info in key_frame_annotation["objects"]:
            if info["id"] in self.node_ids:
                nodes.append(ObjectNode(
                    pos=info["pos"],
                    color=info["color"],
                    speed=info["speed"],
                    heading=info["heading"],
                    id=info['id'],
                    bbox=info['bbox'],
                    height=info['height'],
                    type=info['type'],
                    lane=info['lane'],
                    visible=info['visible'],
                    states=info["states"],
                    collisions=info["collisions"])
                )
        key_graph = SceneGraph(
            ego_id=ego_id, nodes=nodes, folder=self.framepaths[self.idx_key_frame]
        )
        return key_graph

    def generate_statistics(self):
        colors, types, actions, passive_deeds, active_deeds = set(), set(), set(), set(), set()
        for node in self.nodes.values():
            colors.add(node.color)
            types.add(node.type)
            for action in node.actions:
                actions.add(action)
            for interaction in node.interactions.keys():
                if interaction in ["followed", "passed_by", "headed_toward", "accompanied_by"]:
                    passive_deeds.add(interaction)
                else:
                    active_deeds.add(interaction)
        return {
            "<p>": list(colors),
            "<t>": list(types),
            "<s>": list(actions),
            "<passive_deed>": list(passive_deeds),
            "<active_deed>": list(active_deeds)
        }

    def export_trajectories(self, ego=False):
        from vqa.object_node import transform
        center_traj = {}
        bbox_traj = {}
        heading_traj = {}
        origin = self.get_ego_node().positions[0]
        positive_x = self.get_ego_node().headings[0]
        for id in self.nodes.keys():

            assert len(self.nodes[id].positions) == len(self.frames)
            if ego:
                transformed_positions = []
                for position in self.nodes[id].positions:
                    transformed_positions.append(
                        transform_vec(origin, positive_x,
                                      [position])[0]
                    )
                transformed_bboxes = []
                for bbox in self.nodes[id].bboxes:
                    transformed_bboxes.append(
                        transform_vec(origin, positive_x,
                                      bbox)
                    )
                transformed_headings = []
                for heading in self.nodes[id].headings:
                    transformed_headings.append(
                        transform_heading(
                            heading, origin, positive_x
                        )
                    )
                center_traj[id] = transformed_positions  # self.nodes[id].positions
                bbox_traj[id] = transformed_bboxes
                heading_traj[id] = transformed_headings
            else:
                center_traj[id] = self.nodes[id].positions  # self.nodes[id].positions
                bbox_traj[id] = self.nodes[id].bboxes
                heading_traj[id] = self.nodes[id].headings
        json.dump(center_traj, open(f"center.json", "w"))
        json.dump(bbox_traj, open(f"bbox.json", "w"))
        json.dump(heading_traj, open(f"heading.json", "w"))


if __name__ == "__main__":
    EPISODE = "C:/school/Bolei/Merging/MetaVQA/test_collision/0_40_69/**/world*.json"
    import glob

    episode_folder = EPISODE
    # episode_folder = "E:/Bolei/MetaVQA/multiview/0_30_54/**/world*.json"
    frame_files = sorted(glob.glob(episode_folder, recursive=True))
    # print(len(frame_files))
    graph = TemporalGraph(frame_files)
    graph.export_trajectories()

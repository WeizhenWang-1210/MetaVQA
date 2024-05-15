from typing import Union
from vqa.scene_graph import TemporalGraph
from vqa.static_question_generator import Tree, Query
from collections import defaultdict
from vqa.grammar import CFG_GRAMMAR, NO_STATE_CFG
import json
import os
import random

from vqa.functionals import is_stationary, count, \
    CountGreater, CountEqual, Identity, locate_wrapper, extract_color, extract_type, extract_color_unique, \
    extract_type_unique, is_turning, identify_speed, identify_heading, identify_head_toward, predict_trajectory, \
    accelerated

from typing import Callable, Dict
from vqa.object_node import transform


class DynamicQuerySpecifier:
    CACHE = {}

    def __init__(self, type: str, template: dict, parameters: Union[dict, None], graph: TemporalGraph, grammar: dict,
                 debug: bool = False, stats: bool = True) -> None:
        self.type = type
        self.template = template
        self.signature = None
        self.graph = graph
        self.key_frame = self.graph.idx_key_frame
        self.debug = debug
        self.grammar = grammar
        self.stats = stats
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = self.instantiate()
        self.statistics = dict(
            types=defaultdict(int),
            pos=[],
            colors=defaultdict(int),
            actions=defaultdict(int),
            interactions=defaultdict(int),
        )

    def instantiate(self):
        parameters = {}
        signatures = [self.type]
        for param in self.template["params"]:
            local = dict(
                en=None,
                prog=None,
                signature=None,
                ans=None
            )
            if param[1] == "o":
                start_symbol = "<o>"
                param_tree = Tree(start_symbol, 4, self.grammar)

                while param_tree.depth <= 2:
                    param_tree = Tree(start_symbol, 4, self.grammar)
            else:
                print("No fucking way I'm executed")
                start_symbol = param
                param_tree = Tree(start_symbol, 4, self.grammar)
            functional = param_tree.build_functional(self.template["constraint"])
            local["en"] = param_tree.translate()
            # param_tree.visualize()
            signature_string = json.dumps(functional, sort_keys=True)
            local["signature"] = signature_string
            signatures.append(signature_string)

            program = Query([Tree.build_program(functional)], "stuff", Identity,
                            candidates=[node for node in self.graph.get_nodes().values() if
                                        node.id != self.graph.ego_id])  # if node.id != self.graph.ego_id
            local["prog"] = program
            # print(program.heads)
            parameters[param] = local
        self.signature = "-".join(signatures)
        return parameters

    def translate(self, constraint_string=None) -> str:
        variant = random.choice(self.template["text"])
        for param, info in self.parameters.items():
            variant = variant.replace(param, info["en"])
            if constraint_string is not None:
                final_string = "I'm referring to the one that is {}.".format(constraint_string)
                variant = " ".join([variant, final_string])

        return variant

    def find_end_filter(self, string) -> Callable:
        ego_node = self.graph.get_ego_node()
        mapping = dict(
            count=count,
            locate=locate_wrapper(ego_node),
            count_equal=CountEqual,
            count_more=CountGreater,
            extract_color=extract_color,
            extract_color_unique=extract_color_unique,
            extract_type=extract_type,
            extract_type_unique=extract_type_unique,
            is_stationary=is_stationary,
            is_turning=is_turning,
            accelerated=accelerated,
            identify_speed=identify_speed,
            identify_heading=identify_heading(ego_node.pos, ego_node.heading),
            identify_head_toward=identify_head_toward(ego_node),
            predict_trajectory=predict_trajectory(self.key_frame, ego_node.pos, ego_node.heading)
        )
        return mapping[string]

    def answer(self):
        assert self.parameters is not None, "No parameters"
        param_answers = []
        for param, info in self.parameters.items():
            if info["signature"] in DynamicQuerySpecifier.CACHE.keys():
                answers = DynamicQuerySpecifier.CACHE[info["signature"]]
            else:
                query = info["prog"]
                query.set_reference(self.graph.get_ego_node().heading)
                query.set_egos([self.graph.get_ego_node()])
                answers = query.proceed()
                DynamicQuerySpecifier.CACHE[info["signature"]] = answers

            self.parameters[param]["answer"] = answers
            param_answers.append(answers)
        end_filter = self.find_end_filter(self.template["end_filter"])
        ids = []
        for param_answer in param_answers:
            for obj in param_answer:
                ids.append(obj.id)
        return end_filter(param_answers), ids

    def generate_statistics(self, ids):
        """
        The statistics are based on current frame.
        """
        for id in ids:
            obj = self.graph.get_node(id)
            self.statistics["types"][obj.type] += 1
            self.statistics["pos"] += transform(self.graph.get_ego_node(), [obj.pos])
            for action in obj.actions:
                self.statistics["actions"][action] += 1
            self.statistics["colors"][obj.color] += 1
            for interaction in obj.get_all_interactions():
                self.statistics["interactions"][interaction] += 1

    def export_qa(self):
        def type_token_string_converter(token, form):
            # TODO Unify Nomenclature with Chenda.
            mapping = dict(
                nil=dict(singular="object", plural="objects"),
                Bus=dict(singular="bus", plural="buses"),
                Caravan=dict(singular="caravan", plural="caravans"),
                Coupe=dict(singular="coupe", plural="coupes"),
                FireTruck=dict(singular="fire engine", plural="fire engines"),
                Jeep=dict(singular="jeep", plural="jeeps"),
                Pickup=dict(singular="pickup", plural="pickups"),
                Policecar=dict(singular="police car", plural="policecars"),
                SUV=dict(singular="SUV", plural="SUVs"),
                SchoolBus=dict(singular="school bus", plural="school buses"),
                Sedan=dict(singular="sedan", plural="sedans"),
                SportCar=dict(singular="sports car", plural="sports cars"),
                Truck=dict(singular="truck", plural="trucks"),
                Hatchback=dict(singular="hatchback", plural="hatchbacks"),
            )
            if token in mapping.keys():
                return mapping[token][form]
            else:
                return token.lower()

        if "unique" not in self.template["constraint"]:
            question = self.translate()
            answer, ids = self.answer()
            if self.stats:
                self.generate_statistics(ids)
            if self.type == "type_identification":
                answer = [type_token_string_converter(token, "singular").capitalize() for token in answer]
            return [(question, (ids, answer))]
        else:
            qas = []
            answer, ids = self.answer()
            # assume that for unique-constrained questions, the answer a dictionary of form {obj_id: answer}
            for obj_id, answer in answer.items():
                concrete_location = transform(self.graph.get_ego_node(), [self.graph.get_node(obj_id).pos])[0]
                rounded = [int(concrete_location[0]), int(concrete_location[1])]
                question = self.translate("located at {} ".format(rounded))
                if self.stats:
                    self.generate_statistics([obj_id])
                if self.type == "type_identification_unique":
                    answer = type_token_string_converter(answer, "singular").capitalize()
                qas.append((question, (obj_id, answer)))
            return qas


def sample_tree():
    text_tree = Tree(root="<o>", max_depth=4, grammar=CFG_GRAMMAR)
    text_tree.visualize()
    print(text_tree.build_functional(["unique"]))
    print(text_tree.translate())
    print(text_tree.build_program(text_tree.build_functional(["unique"])))


def try_graph(episode):
    episode_folder = episode  # "C:/school/Bolei/Merging/MetaVQA/verification_multiview/95_210_239/**/world*.json"
    import glob
    # episode_folder = "E:/Bolei/MetaVQA/multiview/0_30_54/**/world*.json"
    frame_files = sorted(glob.glob(episode_folder, recursive=True))
    graph = TemporalGraph(frame_files)
    for node in graph.get_nodes().values():
        print(node.id, node.actions)
    for node_id, node in graph.get_nodes().items():
        print(node.id, node.interactions)


def try_pipeline(episode):
    episode_folder = episode  # "C:/school/Bolei/Merging/MetaVQA/verification_multiview/95_210_239/**/world*.json"
    import glob
    # episode_folder = "E:/Bolei/MetaVQA/multiview/0_30_54/**/world*.json"
    frame_files = sorted(glob.glob(episode_folder, recursive=True))
    graph = TemporalGraph(frame_files)
    print(f"KEY FRAME at{graph.framepaths[graph.idx_key_frame]}")
    print(f"Key frame is {graph.idx_key_frame}")
    print(f"Total frame number {len(graph.frames)}")

    template_path = os.path.join("./vqa", "question_templates.json")
    with open(template_path, "r") as f:
        templates = json.load(f)

    statistics = graph.statistics
    grammar = NO_STATE_CFG
    for lhs, rhs in statistics.items():
        if lhs == "<t>":
            grammar[lhs] = [[item] for item in rhs + ["vehicle"]]
        elif lhs == "<active_deed>" or lhs == "<passive_deed>":
            grammar[lhs] = [[item] for item in rhs]
        elif lhs != "<s>":
            grammar[lhs] = [[item] for item in rhs + ["nil"]]
    remove_key = set()
    for lhs, rhs in grammar.items():
        if len(rhs) == 0:
            remove_key.add(lhs)
    new_grammar = grammar

    for token in remove_key:
        new_grammar.pop(token)
    for token in remove_key:
        for lhs, rules in new_grammar.items():
            new_rule = []
            for rhs in rules:
                if token in rhs:
                    continue
                new_rule.append(rhs)
            new_grammar[lhs] = new_rule
    templates = templates["dynamic"]  # templates["generic"]
    # templates = {
    # "identify_stationary": templates["identify_stationary"]
    # "identify_turning": templates["identify_turning"]
    # "identify_acceleration": templates["identify_acceleration"]
    # "identify_speed": templates["identify_speed"]
    # "identify_heading": templates["identify_heading"]
    # "identify_head_toward": templates["identify_head_toward"]
    # "predict_trajectory": templates["predict_trajectory"]
    # }
    for question_type, specification in templates.items():
        q = DynamicQuerySpecifier(
            type=question_type, template=specification, parameters=None,
            graph=graph, grammar=new_grammar, debug=False, stats=False
        )
        result = q.export_qa()
        while len(result) == 0:  # question_type == "identify_turning" and
            q = DynamicQuerySpecifier(
                type=question_type, template=specification, parameters=None,
                graph=graph, grammar=new_grammar, debug=False, stats=False
            )
            print(q.translate())
            result = q.export_qa()
            print(result)


if __name__ == "__main__":
    # EPISODE = "C:/school/Bolei/Merging/MetaVQA/verification_multiview/95_150_179/**/world*.json"
    EPISODE = "C:/school/Bolei/Merging/MetaVQA/verification_multiview/95_210_239/**/world*.json"
    # try_graph(EPISODE)
    try_pipeline(EPISODE)
    # sample_tree()

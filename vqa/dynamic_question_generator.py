from typing import Union

from nltk import CFG

from vqa.dynamic_filter import TemporalNode, TemporalGraph
from question_generator import Tnode, Tree, SubQuery, Query, CACHE
from collections import defaultdict
from vqa.grammar import CFG_GRAMMAR
import json

from vqa.visualization import generate_highlighted

"""
class DynamicQuerySpecifier:
    def __init__(self, type:str, template: dict, parameters: Union[dict, None], graph: TemporalGraph, grammar: dict,
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
            pos=[]
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
                param_tree = Tree(start_symbol, 6, self.grammar)

                while param_tree.depth <= 2:
                    param_tree = Tree(start_symbol, 6, self.grammar)
            else:
                print("No fucking way I'm executed")
                start_symbol = param
                param_tree = Tree(start_symbol, 6, self.grammar)
            functional = param_tree.build_functional(self.template["constraint"])
            local["en"] = param_tree.translate()
            #param_tree.visualize()
            signature_string = json.dumps(functional, sort_keys=True)
            local["signature"] = signature_string
            signatures.append(signature_string)

            program = Query([Tree.build_program(functional)], "stuff", Identity,
                            candidates=[node for node in self.graph.get_nodes() if node.id != self.graph.ego_id]) #if node.id != self.graph.ego_id
            local["prog"] = program
            #print(program.heads)
            parameters[param] = local
        self.signature = "-".join(signatures)
        return parameters

    def translate(self) -> str:
        variant = random.choice(self.template["text"])
        for param, info in self.parameters.items():
            variant = variant.replace(param, info["en"])
        return variant

    def find_end_filter(self, string) -> Callable:
        mapping = dict(
            count=count,
            locate=locate_wrapper(self.graph.get_ego_node()),
            count_equal=CountEqual,
            count_more=CountGreater,
            extract_color=extract_color,
            extract_type=extract_type

        )
        return mapping[string]

    def answer(self):
        assert self.parameters is not None, "No parameters"
        param_answers = []
        for param, info in self.parameters.items():
            if info["signature"] in CACHE.keys():
                answers = CACHE[info["signature"]]
            else:
                query = info["prog"]
                query.set_reference(self.graph.get_ego_node().heading)
                query.set_egos([self.graph.get_ego_node()])
                answers = query.proceed()
                CACHE[info["signature"]] = answers

            self.parameters[param]["answer"] = answers
            param_answers.append(answers)
        end_filter = self.find_end_filter(self.template["end_filter"])
        if self.debug and len(param_answers[0]) > 0:
            objects = []
            ids = []
            for param_answer in param_answers:
                for obj in param_answer:
                    objects.append(f"{obj.type}:{obj.id}")
                    ids.append(obj.id)
            #print(ids)
            parent_folder = os.path.dirname(self.graph.folder)
            identifier = os.path.basename(parent_folder)
            path_to_mask = os.path.join(parent_folder, f"mask_{identifier}.png")
            path_to_mapping = os.path.join(parent_folder, f"metainformation_{identifier}.json")
            folder = parent_folder
            colors = [(1, 1, 1) for _ in range(len(ids))]
            generate_highlighted(
                path_to_mask,
                path_to_mapping,
                folder,
                ids,
                colors
            )
        if self.stats:
            self.generate_statistics(param_answers)

        return end_filter(param_answers)

    def generate_statistics(self, params):
        for objects in params:
            for obj in objects:
                self.statistics["types"][obj.type] += 1
                self.statistics["pos"] += transform(self.graph.get_ego_node(), [obj.pos])

    def generate_mask(self, id):
        parent_folder = os.path.dirname(self.graph.folder)
        identifier = os.path.basename(parent_folder)
        path_to_mask = os.path.join(parent_folder, f"mask_{id}.png")
        path_to_mapping = os.path.join(parent_folder, f"metainformation_{identifier}.json")
        folder = parent_folder
        ids = []
        param_answers = []
        for param, info in self.parameters.items():
            param_answers.append(self.parameters[param]["answer"])
        for param_answer in param_answers:
            for obj in param_answer:
                ids.append(obj.id)
        colors = [(1, 1, 1) for _ in range(len(ids))]
        generate_highlighted(
            path_to_mask,
            path_to_mapping,
            folder,
            ids,
            colors
        )
"""


def sample_tree():
    text_tree = Tree(root = "<o>", max_depth = 5, grammar =  CFG_GRAMMAR)
    text_tree.visualize()
    print(text_tree.build_functional(["unique"]))
    print(text_tree.translate())



if __name__ == "__main__":
   sample_tree()

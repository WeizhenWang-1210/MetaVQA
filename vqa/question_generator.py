from typing import Any, Iterable, Union, LiteralString, Callable
from vqa.functionals import color_wrapper, type_wrapper, state_wrapper, action_wrapper, pos_wrapper, count, \
    CountGreater, CountEqual, Identity, locate_wrapper, extract_color, extract_type, extract_color_unique, \
    extract_type_unique
from vqa.scene_graph import SceneGraph, EpisodicGraph
from vqa.object_node import ObjectNode
from vqa.grammar import STATIC_GRAMMAR, NO_COLOR_STATIC
import random
import json
import argparse
import os
from vqa.visualization import generate_highlighted
from collections import defaultdict
from vqa.object_node import transform

GRAMMAR = STATIC_GRAMMAR


def is_terminal(token, grammar) -> bool:
    return token not in grammar.keys()


class Tnode:
    def __init__(self, token) -> None:
        # print("called {}".format(token))
        self.token = token
        self.parent = None
        self.key = False
        self.children = None

    def populate(self, depth, grammar) -> None:
        if is_terminal(self.token, grammar) or depth <= 0:
            return
        rules = grammar[self.token]
        rule = random.choice(rules)
        if depth == 1:
            while not all(token not in grammar.keys() for token in rule):
                rule = random.choice(rules)
        children = []
        flag = True
        for token in rule:
            new_node = Tnode(token)
            new_node.parent = self
            flag = flag and is_terminal(token, grammar)
            new_node.populate(depth - 1, grammar)
            children.append(new_node)
        self.key = flag
        self.children = children

    def __str__(self) -> str:
        return self.token

    def visualize(self) -> LiteralString | str:
        raise DeprecationWarning
        result = []
        if self.children == None:
            return self.token
        for child in self.children:
            result.append(child.visualize())
        stuff = ",".join(result)
        return "({}_{} : [{}])".format(self.token, self.key, stuff)


class Tree:
    def __init__(self, root: str, max_depth, grammar) -> None:
        self.root = Tnode(root)
        self.grammar = grammar
        self.root.populate(max_depth, self.grammar)
        self.depth = self.get_depth()
        self.functional = None
        self.constriants = None

    def get_depth(self) -> int:
        def depth(node):
            if not node:
                return 0
            if not node.children:
                return 1
            return 1 + max([depth(child) for child in node.children])

        return depth(self.root)

    def build_functional(self, constraints) -> dict:
        self.constriants = constraints
        unique_flag = "unique" in constraints
        """no_color_flag = "no_color" in constraints
        if no_color_flag:
            self.grammar = NO_COLOR_STATIC
            print("here")"""
        results = dict()

        def recur_build_o(root_o, idx):
            assert root_o.token == "<o>" or root_o.token == "<ox>"
            if root_o.key:
                results[idx] = ["us"]
            else:
                curr_cfg = dict(unique=unique_flag)
                offset = 1
                for child in root_o.children:
                    if child.key:
                        curr_cfg[child.token] = child.children[0].token
                    else:
                        expand = dict()
                        for child_child in child.children:
                            if child_child.token == "<o>" or child_child.token == "<ox>":
                                expand["<o>'s id"] = recur_build_o(child_child, idx + offset)
                                offset += 1
                            else:
                                expand[child_child.token] = child_child.children[0].token
                        curr_cfg[child.token] = expand
                results[idx] = curr_cfg
            return idx

        recur_build_o(self.root, 0)
        self.functional = results
        return results

    @classmethod
    def computation_graph(cls, configs):
        def haschild(config):
            result = []
            if not isinstance(config, dict):
                return result
            for key, value in config.items():
                if key == "<o>'s id":
                    result.append(value)
                elif isinstance(value, dict):
                    result += haschild(value)
            return result

        result = []
        for key, value in configs.items():
            ret = haschild(value)
            result += [(key, child) for child in ret]
        return result

    @classmethod
    def build_program(cls, configs):
        def converter(subquery: SubQuery, config: dict, subqueries):
            if isinstance(config, list):
                subquery.us = True
                return
            if "<p>" in config.keys():
                subquery.color = [config["<p>"]] if config["<p>"] != "nil" else None
            elif "<px>" in config.keys():
                subquery.color = [config["<px>"]] if config["<px>"] != "nil" else None
            else:
                print("Major fucked up")
                exit()
            if "<t>" in config.keys():
                subquery.type = [config["<t>"]] if config["<t>"] != "nil" else None
            elif "<tx>" in config.keys():
                subquery.type = [config["<tx>"]] if config["<tx>"] != "nil" else None
            else:
                print("Major fucked up")
                exit()

            subquery.state = [config["<s>"]] if config["<s>"] != "nil" else None
            # Create the action requirements according to the grammar tree specification
            if isinstance(config["<a>"], dict):
                if "<deed_without_o>" in config["<a>"].keys():
                    subquery.action = [config["<a>"]["<deed_without_o>"]]
                else:
                    subquery.action = [config["<a>"]["<deed_with_o>"]]
                    subquery.prev["action"] = subqueries[config["<a>"]["<o>'s id"]]
                    subqueries[config["<a>"]["<o>'s id"]].next = subquery
            else:
                subquery.action = None
            # Create the positional requirements according to the grammar tree specification
            if isinstance(config["<dir>"], dict):
                subquery.pos = [config["<dir>"]["<tdir>"]]
                subquery.prev["pos"] = subqueries[config["<dir>"]["<o>'s id"]]
                subqueries[config["<dir>"]["<o>'s id"]].next = subquery
            else:
                subquery.pos = None

        subqueries = {id: SubQuery() for id in configs.keys()}
        for id, subquery in subqueries.items():
            converter(subquery, configs[id], subqueries)
        root = None
        for subquery in subqueries.values():
            if not subquery.next:
                root = subquery
        return root

    def visualize(self) -> None:
        def better_visualize_tree(node, prefix=""):
            """Visualizes the tree structure."""
            if node is None:
                return
            # Check if the current node is the last child of its parent
            is_last = not node.parent or (node.parent.children and node == node.parent.children[-1])
            # Prepare the prefix for the current level
            current_prefix = prefix + ("└── " if is_last else "├── ")
            # Print the current node
            print(f"{current_prefix}{node.token} (Key: {node.key})")
            # Prepare the prefix for the next level
            next_prefix = prefix + ("    " if is_last else "│   ")
            # Recursively call for each child
            if node.children:
                for child in node.children:
                    better_visualize_tree(child, next_prefix)

        node = self.root
        better_visualize_tree(node)

    def translate(self, object_dict=None):
        if not object_dict:
            if self.functional:
                object_dict = self.functional
            else:
                print('Give me the functional')
                exit()

        def color_token_string_converter(token):
            if token != "nil":
                return token.lower()
            else:
                return ""

        def type_token_string_converter(token, form):
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

        def state_token_string_converter(token):
            if token == "visible" or token == "nil":
                return ""
            return token

        def action_token_string_converter(token, form):
            map = {
                "follow": dict(singular="follows", plural="follow"),
                "pass by": dict(singular="passes by", plural="pass by"),
                "collide with": dict(singular="collides with", plural="collide with"),
                "head toward": dict(singular="heads toward", plural="head toward"),
                "drive alongside": dict(singular="drives alongside", plural="drive alongside"),
                "nil": dict(singular="", plural=""),
                "turn left": dict(singular="turns left", plural="turn left"),
                "turn right": dict(singular="turns right", plural="turn right"),

            }
            return map[token][form]

        def recur_translate(obj_id):
            if len(object_dict[obj_id]) == 1:
                return object_dict[obj_id][0]
            else:
                form = "singular" if object_dict[obj_id]["unique"] else "plural"
                s = state_token_string_converter(object_dict[obj_id]['<s>'])
                if "<p>" in object_dict[obj_id].keys():
                    p = color_token_string_converter(object_dict[obj_id]['<p>'])
                elif "<px>" in object_dict[obj_id].keys():
                    p = color_token_string_converter(object_dict[obj_id]['<px>'])
                else:
                    print("Something terribly wrong happend")
                    exit()
                if "<t>" in object_dict[obj_id].keys():
                    t = type_token_string_converter(object_dict[obj_id]['<t>'], form)
                elif "<tx>" in object_dict[obj_id].keys():
                    t = type_token_string_converter(object_dict[obj_id]['<tx>'], form)
                else:
                    print("Something terribly wrong happened")
                    exit()

                dir = ''
                if isinstance(object_dict[obj_id]['<dir>'], str):
                    dir = object_dict[obj_id]['<dir>'] if object_dict[obj_id]['<dir>'] != 'nil' else ''
                elif isinstance(object_dict[obj_id]['<dir>'], dict):
                    tdir = object_dict[obj_id]['<dir>']['<tdir>']
                    tdir_mapping = {
                        "l": "to the left of",
                        "r": "to the right of",
                        "f": "in front of",
                        "b": "behind",
                        "lf": "to the left and in front of",
                        "rf": "to the right and in front of",
                        "lb": "to the left and behind",
                        "rb": "to the right and behind"
                    }
                    new_o = recur_translate(object_dict[obj_id]['<dir>']["<o>'s id"])
                    dir = tdir_mapping[tdir] + ' ' + new_o
                else:
                    print("warning!")
                a = ''
                if isinstance(object_dict[obj_id]['<a>'], str):
                    a = action_token_string_converter(object_dict[obj_id]['<a>'], form)
                elif isinstance(object_dict[obj_id]['<a>'], dict):
                    if '<deed_with_o>' in object_dict[obj_id]['<a>'].keys():
                        deed_with_o = action_token_string_converter(object_dict[obj_id]['<a>']['<deed_with_o>'], form)
                        new_o = recur_translate(object_dict[obj_id]['<a>']["<o>'s id"])
                        a = deed_with_o + ' ' + new_o
                    else:
                        a = action_token_string_converter(object_dict[obj_id]['<a>']['<deed_without_o>'], form)
                else:
                    print("warning!")
                result = ""
                if form == "singular":
                    result = "the "
                if s != '':
                    result += s + ' '
                if p != '':
                    result += p + ' '
                if t != '':
                    result += t + ' '
                if dir != '':
                    result += dir + ' '
                if a != '':
                    result += 'that ' + a
                result = " ".join(result.split())
                return result

        return recur_translate(0)


class SubQuery:
    """
    A subquery is a single functional program. It can be stringed into a query graph(abstracted in Query). It takes a search space and returns all
    ObjectNode(from the search space) satisfy some conditions(color, type, position w.r.t. to user defined center).
    """

    def __init__(self, color: Iterable[str] = None,
                 type: Iterable[str] = None,
                 pos: Iterable[str] = None,
                 state: Iterable[str] = None,
                 action: Iterable[str] = None,
                 next=None,
                 prev=None) -> None:
        '''
        Initializer
        '''
        self.us = False
        self.state = state  # The state we are looking for

        map = {
            "follow": "follow",
            "pass by": "pass_by",
            "collide with": "collides",
            "head toward": "head_toward",
            "drive alongside": "drive_alongside",
            "nil": None,
            "turn left": "turn_left",
            "turn right": "turn_right",

        }
        self.action = action
        if action:
            self.action = [map[a] for a in
                           action]  # The action we are looking for, with or without respect to some objects
        self.color = color  # The color we are looking for
        self.type = type  # The type we are looking for
        self.pos = pos  # The spatial relationship we are looking for
        self.next = next  # The previous subquery. It's used to retrieve the search space for the candidate
        self.prev = prev if prev else {}  # The next subquery
        self.funcs = None  # The actual functions used to do filtering.
        self.ans = None  # recording the answer in previous call

    def instantiate(self, egos: Iterable[ObjectNode],
                    ref_heading: tuple):
        '''
        Initialize the functions for filtering
        Why delayed insantiatiation? For a subquery in a computation graph, how you instantiate pos functions
        and action functions are dependent of previous question's answers.
        '''
        if self.us:
            # If the query is "us", just return the ego node. No actual search needs to be performed.
            self.funcs = [lambda x: egos]
        else:
            # If this is, indeed, a valid subquery, then create the proper functions to narrow down the search space
            color_func = color_wrapper(self.color) if self.color else None
            type_func = type_wrapper(self.type) if self.type else None
            state_func = state_wrapper(self.state) if self.state else None
            pos_func = pos_wrapper(self.prev["pos"].ans, self.pos, ref_heading) if self.pos else None
            if self.action:
                # TODO Why we need this distinction?
                if "action" not in self.prev.keys():
                    action_func = action_wrapper(egos, self.action)
                else:
                    action_func = action_wrapper(self.prev["action"].ans, self.action)
            else:
                action_func = None
            self.funcs = [color_func, type_func, pos_func, state_func, action_func]

    def __call__(self,
                 candidates: Iterable[ObjectNode],
                 all_nodes: Iterable[ObjectNode]) -> Any:
        """
        candidates: the search space
        all_nodes: the node space
        Use subQuery as an actual function.
        """
        ans = candidates
        if self.us:
            self.ans = self.funcs[0](ans)
            return self.ans
        """
        The tricky part here is that when we keep the entire node space in mind when we do the filtering based on the spatial relationships,
        and consecutive filterings on types and colors are selected from all such nodes.
        """
        # TODO then, is candidates really necessary?
        """
         If self.us, then we always return the ego of the graph. Else, we always use all_nodes to to the position filtering.
        """
        color_func, type_func, pos_func, state_func, action_func = self.funcs
        ans = pos_func(all_nodes) if pos_func else ans
        ans = type_func(ans) if type_func else ans
        ans = color_func(ans) if color_func else ans
        ans = state_func(ans) if state_func else ans
        ans = action_func(ans) if action_func else ans
        self.ans = ans
        return self.ans

    def __str__(self) -> str:
        # print(self.format,[node.id.split("-")[0] for node in self.ans])
        dict = {}
        for path, prog in self.prev.items():
            dict[path] = prog.__str__()
        my = {
            "color": self.color,
            "type": self.type,
            "pos": self.pos,
            "prev": dict,
            "ego": self.us,
            "funcs": [f is not None for f in self.funcs] if self.funcs else self.funcs,
            "ans": self.ans
        }
        return json.dumps(my)


class Query:
    """
    A query is a functional implentation of an English question. It can have a single-thread of subqueries(counting/referral),
    (in fact, better yet, we can abstrac the aggregat method of single-thread as the "identity" method: extracting the one-and-only answer)
    or it can have multiple lines of subqueries and aggregate the answers at the end.(In logical questions). 
    self.ans should either be None or bboxes (n, 4, 2), int, True/False
    """

    def __init__(self, heads: Iterable[SubQuery], format: str, end_filter: callable,
                 ref_heading: tuple = None,
                 candidates: Iterable[ObjectNode] = None,
                 ) -> None:
        '''
        heads: the beginning subqueries of our query. Guaranteed linked-list-like structure. Reach the end and aggregate the result with end_filter
        format: a str for specifying what kind of question this is
        end_filter: a function: Iterable[Iterable] -> Iterable[]. Used to aggregate answers
        ref_heading: a vector that indicates the direction of positive x-coordinate, in world coordinate
        candidates: the search space of the Query
        '''
        self.candidates = candidates
        self.format = format
        self.heads = heads
        self.final = end_filter
        self.ref_heading = ref_heading
        self.ans = None
        self.egos = None

    def set_reference(self, heading: tuple) -> None:
        '''
        set reference heading
        '''
        self.ref_heading = heading

    def set_searchspace(self, nodes: Iterable[ObjectNode]) -> None:
        '''
        set the allowed searchspace
        '''
        self.candidates = nodes

    def set_egos(self, nodes: Iterable[ObjectNode]):
        '''
        set the center vehicles
        '''
        self.egos = nodes

    def proceed(self):
        '''
        Go down the query, store partial answers in sub-query, store the final answer in query
        '''

        def postorder_traversal(subquery, egos, ref_heading, all_nodes):
            """
            Return: A list of nodes being the answer of the previous question.
            """

            if subquery.prev:
                for child_subquery in subquery.prev.values():
                    postorder_traversal(child_subquery, egos, ref_heading, all_nodes)
            if not subquery.funcs:
                subquery.instantiate(egos, ref_heading)
            result = all_nodes
            result = subquery(result, all_nodes)

        search_spaces = []
        for root in self.heads:
            postorder_traversal(root, self.egos, self.ref_heading, self.candidates)
            search_spaces.append(root.ans)
        self.ans = self.final(search_spaces)
        return self.ans


CACHE = dict()


class QuerySpecifier:
    def __init__(self, type: str, template: dict, parameters: Union[dict, None], graph: SceneGraph, grammar: dict,
                 debug: bool = False, stats: bool = True) -> None:
        self.type = type
        self.template = template
        self.signature = None
        self.graph = graph
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
                            candidates=[node for node in self.graph.get_nodes() if
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
                final_string = "I'm referring to one that is {}.".format(constraint_string)
                variant = " ".join([variant, final_string])

        return variant

    def find_end_filter(self, string) -> Callable:
        mapping = dict(
            count=count,
            locate=locate_wrapper(self.graph.get_ego_node()),
            count_equal=CountEqual,
            count_more=CountGreater,
            extract_color=extract_color,
            extract_color_unique=extract_color_unique,
            extract_type=extract_type,
            extract_type_unique=extract_type_unique,

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
                # query.set_searchspace(self.graph.get_nodes())
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
            # print(ids)
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
        """
        Create a mask corresponding to the answer of a problem

        """
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

    def export_qa(self):
        def type_token_string_converter(token, form):
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
            answer = self.answer()
            if self.type == "type_identification":
                answer = [type_token_string_converter(token, "singular").capitalize() for token in answer]
            return [(question, answer)]
        else:
            qas = []
            answer = self.answer()
            # assume that for unique-constrained questions, the answer a dictionary of form {obj_id: answer}
            for obj_id, answer in answer.items():
                concrete_location = transform(self.graph.get_ego_node(), [self.graph.get_node(obj_id).pos])[0]
                rounded = (int(concrete_location[0]), int(concrete_location[1]))
                question = self.translate("located at {} ".format(
                        rounded
                    )
                )
                if self.type == "type_identification_unique":
                    answer = type_token_string_converter(answer, "singular").capitalize()
                qas.append((question, (obj_id,answer)))
            return qas


def main():
    # pwd = os.getcwd()
    # absolute_file_path = os.path.abspath(__file__)
    # template_path = os.path.join(pwd, "vqa/question_templates.json")
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the other file
    template_path = os.path.join(current_directory, "question_templates.json")
    with open(template_path, "r") as f:
        templates = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="some/5_80_119/5_100/world_5_100")
    parser.add_argument("--episode", type=str, default="verification/0_30_49")
    args = parser.parse_args()

    try:
        # print(args.step)
        with open('{}.json'.format(args.step), 'r') as scene_file:
            scene_dict = json.load(scene_file)
    except Exception as e:
        raise e
    from vqa.scene_graph import nodify
    agent_id, nodes = nodify(scene_dict)
    graph = SceneGraph(agent_id, nodes, folder="some/5_80_119/5_100/world_5_100")
    # episode_graph = EpisodicGraph()
    # frames = [f for f in os.listdir(args.episode) if not ("." in f)]
    # interaction_path = os.path.join(args.episode, "interaction.json")
    # episode_graph.load(args.episode, frames, interaction_path)
    # graph = episode_graph.final_frame
    GRAMMAR = STATIC_GRAMMAR
    """
    q1 = SubQuery(
        color= None,
        type = ["dog"], 
        pos= ['rf'], 
        state= None, 
        action= None,
        next = None,
        prev = {})    
    q2 = SubQuery()
    q2.us = True
    q1.prev["pos"] = q2
    q2.next = q1
    parameters = {
        "<o>":{
            "en": "dog to the right and in front of us",
            "prog": Query([q1],"counting",Identity),
            "ans":None
        }
    }

    q3 = SubQuery(
        color= None,
        type = ["vehicle"], 
        pos= ["lf"], 
        state= None, 
        action= None,
        next = None,
        prev = {})    
    q4 = SubQuery()
    q4.us = True
    q3.prev["pos"] = q4
    q4.next = q3

    q5 = SubQuery(
        color= ["White"],
        type = ["vehicle"], 
        pos= None, 
        state= None, 
        action= None,
        next = None,
        prev = {})

    parameters_1 = {
        "<o>":{
            "en": "dog to the right and in front of us",
            "prog": Query([q1],"counting",Identity),
            "ans":None
        }
    }
    parameters_2 = {
        "<o>":{
            "en": "vehicles to the left and in front of us",
            "prog": Query([q3],"counting",Identity),
            "ans":None
        }
    }

    parameters_3 = {
        "<o>":{
            "en": "white vehicle",
            "prog": Query([q5],"counting",Identity),
            "ans":None
        }
    }

    q = QuerySpecifier(template=templates["generic"]["counting"], parameters = parameters_3,graph = graph)
    print(q.translate())
    print(q.answer())


    prophet = QueryAnswerer(graph,[q])
    result = prophet.ans(q)
    print(result[0].id)

    ids = [node.id for node in  q.parameters["<o>"]["answer"] if node.id != agent_id]
    print(len(ids))
    from vqa.visualization import generate_highlighted
    generate_highlighted(path_to_mask =  "verification/10_299/mask_10_299.png",
                         path_to_mapping= "verification/10_299/metainformation_10_299.json",
                         folder = "verification/10_299",
                         ids = ids,
                         colors = [(1,1,1)]*len(ids))"""

    for lhs, rhs in graph.statistics.items():
        # GRAMMAR[lhs] = [[item] for item in rhs + ['nil']]
        if lhs == "<p>":
            GRAMMAR[lhs] = [[item] for item in rhs + ["nil"]]
        else:
            GRAMMAR[lhs] = [[item] for item in rhs]

    q3 = SubQuery(
        color=None,  # ["Blue"],
        type=["vehicle"],
        pos=None,  # ["rf"],
        state=None,
        action=None,
        next=None,
        prev={})
    # q4 = SubQuery()
    # q4.us = True
    # q3.prev["pos"] = q4
    # q4.next = q3
    print(q3)
    parameters_2 = {
        "<o>": {
            "en": "Pedestrian",
            "prog": Query([q3], "counting", Identity,
                          candidates=[node for node in graph.get_nodes() if node.id != agent_id]),
            "ans": None,
            "signature": "",
        }
    }
    q = QuerySpecifier(
        type="counting",
        template=templates["generic"]["counting"],
        parameters=parameters_2,
        graph=graph,
        grammar=GRAMMAR,
        debug=True,
    )

    print(q.translate())
    print(q.answer())
    print(q.parameters)
    print(q.statistics)
    print("end")


def some_tree(file):
    with open(file, 'r') as scene_file:
        scene_dict = json.load(scene_file)
    from vqa.scene_graph import nodify
    agent_id, nodes = nodify(scene_dict)
    graph = SceneGraph(agent_id, nodes, folder=file)
    grammar = GRAMMAR
    for lhs, rhs in graph.statistics.items():
        if lhs == "<p>":
            grammar[lhs] = [[item] for item in ["nil"]]
        else:
            grammar[lhs] = [[item] for item in ["vehicle"] + rhs]
    tree = Tree("<o>", 2, grammar)
    tree.visualize()
    plan = tree.build_functional(["no_color"])
    prog = Tree.build_program(plan)
    print(prog)
    query = Query([prog], "stuff",
                  Identity,
                  ref_heading=graph.get_ego_node().heading,
                  candidates=[node for node in graph.get_nodes()])
    query.set_egos([graph.get_ego_node()])
    answer = query.proceed()
    print(answer)
    generate_highlighted('SOME.png', "verification/9_41_80/9_41/metainformation_5_100.json",
                         )


if __name__ == "__main__":
    # print("hello")
    # main()
    some_tree("verification/9_41_80/9_41/world_9_41.json")

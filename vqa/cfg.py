import json
import random
from itertools import product
from vqa.grammar import CFG_GRAMMAR
from vqa.question_generator import SubQuery
from pprint import pprint

def is_terminal(token)->bool:
    return token not in CFG_GRAMMAR.keys()


class tnode:
    def __init__(self,token) -> None:
        #print("called {}".format(token))
        self.token = token
        self.parent = None
        self.key = False
        self.children = None
    
    def populate(self, depth):
        if is_terminal(self.token) or depth <= 0:
            return
        rules = CFG_GRAMMAR[self.token]
        rule = random.choice(rules)
        if depth == 1:
            while not all(token not in CFG_GRAMMAR.keys() for token in rule):
                rule = random.choice(rules)
        children = []
        flag = True
        for token in rule:
            new_node = tnode(token)
            new_node.parent = self
            flag = flag and is_terminal(token)
            new_node.populate(depth - 1)
            children.append(new_node)
        self.key = flag
        self.children = children
    
    def __str__(self) -> str:
        return self.token

    def visualize(self):
        result = []
        if self.children == None:
            return self.token
        for child in self.children:
            result.append(child.visualize())
        stuff = ",".join(result)
        return "({}_{} : [{}])".format(self.token, self.key,stuff)

        



class tree:
    def __init__(self, max_depth) -> None:
        self.root = tnode("<o>")
        self.root.populate(max_depth)
        self.depth = self.get_depth()
    def get_depth(self)->int:
        def depth(node):
            if not node:
                return 0
            if not node.children:
                return 1
            return 1 + max([depth(child) for child in node.children])
        return depth(self.root)
    def new_build_functional(self,constraints):
        unique_flag = "unique" in constraints
        results = dict()
        def recur_build_o(root_o, id):
            assert root_o.token == "<o>"
            if root_o.key:
                results[id] = ["us"]
            else:
                curr_cfg = dict(unique = unique_flag)
                offset = 1
                for child in root_o.children:
                    if child.key:
                        curr_cfg[child.token] = child.children[0].token
                    else:
                        expand = dict()
                        for child_child in child.children:
                            if child_child.token == "<o>":
                                expand["<o>'s id"] = recur_build_o(child_child, id + offset)
                                offset += 1
                            else:
                                expand[child_child.token] = child_child.children[0].token
                        curr_cfg[child.token] = expand
                results[id] = curr_cfg
            return id
        recur_build_o(self.root, 0)
        return results


    def computation_graph(self, configs):
        def haschild(config):
            result = []
            if not isinstance(config,dict):
                return result
            for key, value in config.items():
                if key == "<o>'s id":
                    result.append(value)
                elif isinstance(value, dict):
                    result += haschild(value)
            return result
        result = []
        for key,value in configs.items():
            ret = haschild(value)
            result += [(key, child) for child in ret]
        return result
    
    def build_program(self, configs):
        def converter(subquery, config, subqueries):
            if isinstance(config, list):
                subquery.us = True
                return
            subquery.color = [config["<p>"]],
            subquery.type = [config["<t>"]],
            subquery.state = [config["<s>"]],             
            #Create the action requirements according to the grammar tree specification
            if isinstance(config["<a>"],dict):
                if "<deed_without_o>" in config["<a>"].keys():
                    subquery.action = [config["<a>"]["<deed_without_o>"]]
                else:
                    subquery.action =  [config["<a>"]["<deed_with_o>"]]
                    if subquery.prev == None:
                        subquery.prev = {"action": subqueries[config["<a>"]["<o>'s id"]]}
                    else:
                        subquery.prev.action = subqueries[config["<a>"]["<o>'s id"]]
                    subqueries[config["<a>"]["<o>'s id"]].next = subquery
            else:
                subquery.action = None
            #Create the positional requirements according to the grammar tree specification
            if isinstance(config["<dir>"],dict):
                subquery.pos = [config["<dir>"]["<tdir>"]]
                if subquery.prev == None:
                    subquery.prev = {"pos": subqueries[config["<dir>"]["<o>'s id"]]}
                else:
                    subquery.prev.pos = subqueries[config["<dir>"]["<o>'s id"]]
                subqueries[config["<dir>"]["<o>'s id"]].next = subquery
            else:
                subquery.pos = None
        subqueries = {id: SubQuery() for id in configs.keys()}
        prev = None
        for id, item in subqueries.items():
            item.prev = prev
            prev = item
        for id, subquery in subqueries.items():
            converter(subquery, configs[id], subqueries)
        root = None
        for subquery in subqueries.values():
            if not subquery.next:
                root = subquery
        return root
    
    """def build_Query(self, ref_heading, end_filter)"""
        
            


    
        
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
        
def translate(object_dict):
    def recur_translate(obj_id):
        if len(object_dict[obj_id]) == 1:
            return object_dict[obj_id][0]
        else:
            form = "singular" if object_dict[obj_id]["unique"] else "plural"
            s = state_token_string_converter(object_dict[obj_id]['<s>'])
            p = color_token_string_converter(object_dict[obj_id]['<p>']) 
            t = type_token_string_converter(object_dict[obj_id]['<t>'],form)
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
                a = action_token_string_converter(object_dict[obj_id]['<a>'],form)
            elif isinstance(object_dict[obj_id]['<a>'], dict):
                if '<deed_with_o>' in object_dict[obj_id]['<a>'].keys():
                    deed_with_o = action_token_string_converter(object_dict[obj_id]['<a>']['<deed_with_o>'],form)
                    new_o = recur_translate(object_dict[obj_id]['<a>']["<o>'s id"])
                    a = deed_with_o + ' ' + new_o
                else:
                    a = action_token_string_converter(object_dict[obj_id]['<a>']['<deed_without_o>'],form)
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


def color_token_string_converter(token):
    if token != "nil":
        return token.lower()
    else:
        return ""

def type_token_string_converter(token,form):
    mapping = dict(
        nil = dict(singular = "thing", plural = "things"),
        Bus = dict(singular = "bus", plural = "buses"),
        Caravan = dict(singular = "caravan", plural = "caravans"),
        Coupe = dict(singular = "coupe", plural = "coupes"),
        FireTruck = dict(singular = "fire engine", plural = "fire engines"),
        Jeep = dict(singular = "jeep", plural = "jeeps"),
        Pickup = dict(singular = "pickup", plural = "pickups"),
        Policecar = dict(singular = "police car", plural = "policecars"),
        SUV = dict(singular = "SUV", plural = "SUVs"),
        SchoolBus = dict(singular = "school bus", plural = "school buses"),
        Sedan = dict(singular = "sedan", plural = "sedans"),
        SportCar = dict(singular = "sports car", plural = "sports cars"),
        Truck = dict(singular = "truck", plural = "trucks"),
        Hatchback = dict(singular = "hatchback", plural = "hatchbacks")
    )
    return mapping[token][form]

def state_token_string_converter(token):
    if token == "visible" or token == "nil":
        return ""
    return token

def action_token_string_converter(token, form):
    map = {
        "follow" : dict(singular = "follows", plural = "follow"),
        "pass by": dict(singular = "passes by", plural = "pass by"),
        "collide with": dict(singular = "collides with", plural = "collide with"),
        "head toward":dict(singular = "heads toward", plural = "head toward"),
        "drive alongside": dict(singular = "drives alongside", plural = "drive alongside"),
        "nil": dict(singular = "", plural = ""),
        "turn left" : dict(singular = "turns left", plural = "turn left"),
        "turn right": dict(singular = "turns right", plural = "turn right"),

    }
    return map[token][form]
mytree = tree(4)

better_visualize_tree(mytree.root)
FUNCTIONALS = mytree.new_build_functional([])
print(FUNCTIONALS)
print(translate(FUNCTIONALS))
print(mytree.computation_graph(FUNCTIONALS))
root = mytree.build_program(FUNCTIONALS)
print(root.prev)
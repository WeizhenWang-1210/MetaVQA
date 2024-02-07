from typing import Any, Callable, Iterable, List
from vqa.scene_graph import SceneGraph
from vqa.object_node import ObjectNode,nodify,transform
from vqa.dataset_utils import extend_bbox
from collections import defaultdict
from vqa.grammar import CFG_GRAMMAR
import random
from vqa.dynamic_filter import DynamicFilter
import json
import argparse
import os
# pwd = os.getcwd()
# absolute_file_path = os.path.abspath(__file__)
# template_path = os.path.join(pwd, "vqa/question_templates.json")
current_directory = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the other file
template_path = os.path.join(current_directory, "question_templates.json")
with open(template_path,"r") as f:
    templates = json.load(f)





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
    def __init__(self, root, max_depth) -> None:
        self.root = tnode(root)
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
    def build_functional(self,constraints):
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
            subquery.color = config["<p>"],
            subquery.type = config["<t>"],
            subquery.state = config["<s>"],             
            #Create the action requirements according to the grammar tree specification
            if isinstance(config["<a>"],dict):
                if "<deed_without_o>" in config["<a>"].keys():
                    subquery.action = [config["<a>"]["<deed_without_o>"]]
                else:
                    subquery.action =  [config["<a>"]["<deed_with_o>"]]
                    subquery.prev["action"] = subqueries[config["<a>"]["<o>'s id"]]
                    subqueries[config["<a>"]["<o>'s id"]].next = subquery
            else:
                subquery.action = None
            #Create the positional requirements according to the grammar tree specification
            if isinstance(config["<dir>"],dict):
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
    def visualize(self):
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
    def translate(self, object_id)

        
def translate(object_dict):
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



class SubQuery:
    """
    A subquery is a single functional program. It can be stringed into a query graph(abstracted in Query). It takes a search space and returns all
    ObjectNode(from the search space) satisfy some conditions(color, type, position w.r.t. to user defined center).
    """
    def __init__(self, color:Iterable[str] = None, 
                 type:Iterable[str] = None,
                 pos: Iterable[str] = None,
                 state: Iterable[str] = None,
                 action: Iterable[str] = None,
                 next = None,
                 prev = None) -> None:
        '''
        Initializer
        '''
        self.us = False
        self.state = state #The state we are looking for
        self.action = action #The action we are looking for, with or without respect to some objects
        self.color = color #The color we are looking for
        self.type = type   #The type we are looking for
        self.pos = pos     #The spatial relationship we are looking for
        self.next = next   #The previous subquery. It's used to retrieve the search space for the candidate
        self.prev = prev if prev else {}   #The next subquery
        self.funcs = None  #The actual functions used to do filtering.
        self.ans = None    #recording the answer in previous call
    
    def instantiate(self, egos: Iterable[ObjectNode],
                    ref_heading:tuple):
        '''
        Initialize the functions for filtering
        '''
        if self.us:
            #If the query is "us", just return the ego node. No actual search needs to be performed.
            self.funcs = [lambda x: egos]
        else: 
            #If this is, indeed, a valid subquery, then create the proper functions to narrow down the search space
            color_func = color_wrapper(self.color) if self.color else None
            type_func = type_wrapper(self.type) if self.type else None
            state_func = state_wrapper(self.state) if self.state else None
            pos_func = pos_wrapper(self.prev["pos"].ans, self.pos, ref_heading) if self.pos else None
            action_func = action_wrapper(self.prev["action"].ans, self.action) if self.action else None
            self.funcs = [color_func, type_func, pos_func, state_func, action_func]
        
    def __call__(self, 
                 candidates:Iterable[ObjectNode],
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
        color_func, type_func,pos_func,state_func, action_func = self.funcs
        ans = pos_func(all_nodes) if pos_func else ans
        ans = type_func(ans) if type_func else ans
        ans = color_func(ans) if color_func else ans
        ans = state_func(ans) if state_func else ans
        ans = action_func(ans) if action_func else ans
        self.ans = ans
        return self.ans




     

class Query:
    """
    A query is a functional implentation of an English question. It can have a single-thread of subqueries(counting/referral),
    (in fact, better yet, we can abstrac the aggregat method of single-thread as the "identity" method: extracting the one-and-only answer)
    or it can have multiple lines of subqueries and aggregate the answers at the end.(In logical questions). 
    self.ans should either be None or bboxes (n, 4, 2), int, True/False
    """
    def __init__(self, heads: Iterable[SubQuery], format: str, end_filter: callable,
                  ref_heading: tuple = None, 
                  candidates:Iterable[ObjectNode] = None,
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

    def set_reference(self, heading: tuple)->None:
        '''
        set reference heading
        '''
        self.ref_heading = heading
    
    def set_searchspace(self, nodes: Iterable[ObjectNode])->None:
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

        def postorder_traversal(subquery,egos, ref_heading, all_nodes):
            """
            Return: A list of nodes being the answer of the previous question.
            """
            
            if subquery.prev:
                for child_subquery in subquery.prev.values():
                    postorder_traversal(child_subquery, egos, ref_heading, all_nodes)
            if not subquery.funcs:
                subquery.instantiate(egos, ref_heading)
            result = all_nodes
            result = subquery(result,all_nodes)
        
        search_spaces = []
        for root in self.heads:
            postorder_traversal(root,self.egos,self.ref_heading,self.candidates)
            search_spaces.append(root.ans)
        self.ans = self.final(search_spaces)
        return self.ans
    
    def __str__(self) -> str:
        '''
        get all answers
        '''
        #print(self.format,[node.id.split("-")[0] for node in self.ans])
        print(self.ans)
     
"""
Only return when candidates are visible by ego
"""

class QuerySpecifier:
    def __init__(self,template:dict, parameters:dict, graph:SceneGraph) -> None:
        self.template = template
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = self.instantiate()
        self.graph = graph
        
    def instantiate(self):
        parameters = {}
        for param in self.template["params"]:
            local = dict(
                en = None,
                prog = None,
                ans = None
            )
            if param[1]=="o":
                start_symbol = "<o>"
                param_tree = tree(start_symbol,4)
                
                while(param_tree.depth<=2):
                    param_tree = tree(start_symbol,4)
            else:
                start_symbol = param
                param_tree = tree(start_symbol,4)
            functional = param_tree.build_functional(self.template["constraint"])
            local["en"] = translate(functional)
            program = Query([param_tree.build_program(functional)],"stuff",Identity)
            local["prog"] = program
            parameters[param] = local
        return parameters

    def translate(self):
        variant = random.choice(self.template["text"])
        for param, info in self.parameters.items():
            variant = variant.replace(param,info["en"])
        return variant
    
    def find_end_filter(self, string):
        mapping = dict(
            count = count,
            locate = locate,
            count_equal = CountEqual,
            count_more = CountGreater,
        )
        return mapping[string]

    def answer(self):
        param_answers = []
        for param,info in self.parameters.items():
            query = info["prog"]
            query.set_reference(self.graph.get_ego_node().heading)
            query.set_searchspace(self.graph.get_nodes())
            query.set_egos([self.graph.get_ego_node()])
            answers = query.proceed()
            self.parameters[param]["answer"] = answers
            param_answers.append(answers)
        end_filter = self.find_end_filter(self.template["end_filter"])
        return end_filter(param_answers)

           


def color_wrapper(colors:Iterable[str])->Callable:
    '''
    Constructor for a function that return all nodes with color in colors
    '''
    def color(candidates:Iterable[ObjectNode]):
        results = []
        for candidate in candidates:
            if candidate.visible and candidate.color in colors:
                results.append(candidate)
        return results
    return color

def type_wrapper(types:Iterable[str])->Callable:
    '''
    Constructor for a function that return all nodes with type in types or is a subtype of type in types
    '''
    #print(types)
    def type(candidates:Iterable[ObjectNode]):
        #print(candidates)
        if not candidates:
            return []
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            #print(candidate)
            for t in types:
                if candidate.type == t  or subclass(candidate.type, t):
                    results.append(candidate)
                    break
        return results
    return type

def state_wrapper(states:Iterable[str])->Callable:
    '''
    Constructor for a function that return all nodes with one state in states
    '''
    #print(types)
    def state(candidates:Iterable[ObjectNode]):
        #print(candidates)
        if not candidates:
            return []
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            #print(candidate)
            for s in states:
                #print(candidate.type, t)
                if s in candidate.state:
                    #print(candidate.id)
                    results.append(candidate)
                    break
        return results
    return state

def action_wrapper(egos: [ObjectNode], actions: Iterable[str], ref_heading: tuple = None)->Callable:
    scene_folder = "E:/bolei/metavqa/verification"
    filter = DynamicFilter(scene_folder,sample_frequency=1, episode_length=10, skip_length=10)
    filter.process_episodes(sample_frequency=1, episode_length=10, skip_length=10)
    def act(candidates: Iterable[ObjectNode]):
        mapping = {
            "follow": filter.follow,
            "pass by": filter.pass_by,
            "collide with":filter.collide_with,
            "head toward": filter.head_toward,
            "drive alongside": filter.drive_alongside
        }
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            for ego in egos:
               for action in actions:
                   if ego.id != candidate.id and mapping[action](ego, candidate):
                       results.append(candidate)
        return results
    return act


def pos_wrapper(egos: [ObjectNode], spatial_retionships: Iterable[str], ref_heading: tuple = None)->Callable:
    '''
    A constructor for selecting all nodes that exhibit spatial_relationship with any ego in egos for spatial_relationship in spatial_relationships.
    ref_heading is provided to define what's left v.s. right
    '''
    def pos(candidates: Iterable[ObjectNode]):
        results = []
        for candidate in candidates:
            if not candidate.visible:
                continue
            for ego in egos:
                if ego.id != candidate.id and ego.compute_relation_string(candidate, ref_heading) in spatial_retionships:
                    results.append(candidate)
        return results
    return pos

def subclass(class1:str, class2:str)->bool:
    '''
    determine if class1 is the subclass of class2
    '''
    inheritance = get_inheritance() #inheritance is not a tree. But, it's a DAG from supertype to subtype(like your typing system in C++)
    if class1 == class2:
        return True
    result = False
    for child in inheritance[class2]:
        result = result or subclass(class1, child)
    return result

def get_inheritance()->defaultdict:
    '''
    Return a lineage tree as a dictionary
    '''
    import yaml
    with open("./asset_config.yaml","r") as stream:
        tree = yaml.safe_load(stream)["type"]

    inheritance = defaultdict(lambda:[])

    def get_non_leaf_nodes(d, inheritance, parent_key='', ):
        non_leaf_nodes = []
        for key, value in d.items():
            # Construct a full key path if you are in a nested dictionary
            full_key = parent_key + '.' + key if parent_key else key
            if isinstance(value, dict):
                inheritance[parent_key].append(key)
                non_leaf_nodes.append(full_key)
                # Recursively search for non-leaf nodes
                non_leaf_nodes.extend(get_non_leaf_nodes(value, inheritance, key))
        return non_leaf_nodes
    get_non_leaf_nodes(tree, inheritance)
    return inheritance
        


def greater(A,B)->bool:
    '''
    checker
    '''
    return A > B
def equal(A,B)->bool:
    '''
    checker
    '''
    return A==B
def less(A,B)->bool:
    '''
    checker
    '''
    return A<B
def count(stuff: Iterable)->int:
    '''
    checker
    '''
    return [len(s) for s in stuff]
    
'''
End filters
'''

def CountGreater(search_spaces)->bool:
    """
    Return True if the first set in the search_spaces has greater length than the second set.
    """
    assert len(search_spaces) == 2, "CountGreater should have only two sets to work with"
    nums = count(search_spaces)#[count(search_space) for search_space in search_spaces]
    return greater(nums[0],nums[1])

def CountEqual(search_spaces)->bool:
    """
    Return True if all sets in search_spaces have the same length.
    """
    nums = count(search_spaces)#[count(search_space) for search_space in search_spaces]
    first = nums[0]
    for num in nums:
        if num != first:
            return False
    return True

def CountLess(search_spaces)->bool:
    """
    Return True if the first set in the search_spaces has greater smaller than the second set.
    """
    assert len(search_spaces) == 2, "CountGreater should have only two sets to work with"
    nums = count(search_spaces)#[count(search_space) for search_space in search_spaces]
    return less(nums[0],nums[1])

def Describe(search_spaces)->str:
    """
    Return True if the first set in the search_spaces has greater smaller than the second set.
    """
    search_spaces = search_spaces[0]
    if len(search_spaces) == 0:
        return "No, there is not any item with specified action"
    result = "Yes, there is "
    result += search_spaces[0].color
    result += " "
    result += search_spaces[0].type
    if len(search_spaces) == 1:
        return result
    else:
        for node in search_spaces[1:]:
            result += " and "
            result += node.color
            result += " "
            result += node.type
        result += '.'
    return result

def Identity(search_spaces):
    '''
    Return the singleton answer in search spaces
    '''
    return search_spaces[0]

def locate(stuff: Iterable[ObjectNode], origin: ObjectNode)->Iterable:
    """
    Return the bbox of all AgentNodes in stuff.
    """
    result = []
    for s in stuff:
        for more_stuff in s:
            result.append(more_stuff.bbox)
            print(transform(origin,more_stuff.bbox))
    return result

def locate_wrapper(origin: ObjectNode)->Callable:
    """
    The returned function takes in an Iterable of ObjectNode and returns the 3d bounding boxes in ego's coordinate
    """
    def locate(stuff: Iterable[ObjectNode]):
        result = []
        for s in stuff:
            for more_stuff in s:
                ego_bbox = transform(origin,more_stuff.bbox)
                ego_3dbbox = extend_bbox(ego_bbox, more_stuff.height)
                result.append(ego_3dbbox)
        return result
    return locate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default = "verification/10_299/world_10_299")
    args = parser.parse_args()
    try:
        #print(args.step)
        with open('{}.json'.format(args.step),'r') as scene_file:
                scene_dict = json.load(scene_file)
    except Exception as e:
        raise e
    agent_id,nodes = nodify(scene_dict)
    graph = SceneGraph(agent_id,nodes)    

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
    
    """
    prophet = QueryAnswerer(graph,[q])
    result = prophet.ans(q)
    print(result[0].id)
    """
    ids = [node.id for node in  q.parameters["<o>"]["answer"] if node.id != agent_id]
    print(len(ids))
    from vqa.visualization import generate_highlighted
    generate_highlighted(path_to_mask =  "verification/10_299/mask_10_299.png",
                         path_to_mapping= "verification/10_299/metainformation_10_299.json",
                         folder = "verification/10_299",
                         ids = ids,
                         colors = [(1,1,1)]*len(ids))

            
import json
import random
from itertools import product
from vqa.grammar import CFG_GRAMMAR
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
    def new_build_functional(self):
        results = dict()
        def recur_build_o(root_o, id):
            assert root_o.token == "<o>"
            if root_o.key:
                results[id] = ["us"]
            else:
                curr_cfg = dict()
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

    #
    #
    # def new_build_functional(self):
    #     total_key = ["<s>", "<p>", "<t>", "<dir>", "<a>"]
    #     todo_node = [(self.root, None)] # (node, parent), for example, (<o>, dir) means current code suppose to fill previous dir
    #     pending_cfg = []
    #     results = []
    #     unique_identifier = 0
    #     while len(todo_node) > 0:
    #         cur_o, fill_target = todo_node.pop()
    #         curr_cfg = dict()
    #         if cur_o.token == "<o>" and not cur_o.key:
    #             pending_flag = False
    #             curr_cfg["id"] = unique_identifier
    #             unique_identifier += 1
    #             for child in cur_o.children:
    #                 if child.key:
    #                     curr_cfg[child.token] = child.children[0].token
    #                 else:
    #                     pending_flag = True
    #                     for child_child in child.children:
    #                         todo_node.append((child_child, child.token))
    #             if pending_flag:
    #                 for key in total_key:
    #                     if key not in curr_cfg:
    #                         curr_cfg[key] = None
    #                         curr_cfg[key + "_pointer"] = None
    #                 pending_cfg.append(curr_cfg)
    #             else:
    #                 results.append(curr_cfg)
    #                 while self.check_pending_full(pending_cfg, fill_target):
    #                     update_cfg = pending_cfg.pop()
    #                     update_cfg["{}_pointer".format(fill_target)] = curr_cfg['id']
    #                     curr_cfg = update_cfg
    #                     results.append(curr_cfg)
    #         elif cur_o.token == "<o>" and cur_o.key:
    #             curr_cfg["id"] = unique_identifier
    #             unique_identifier += 1
    #         else:
    #             update_cfg = pending_cfg.pop()
    #             update_cfg[cur_o.token] = cur_o.children[0].token
    #             pending_cfg.append(update_cfg)
    #
    # def check_pending_full(self, pending_cfg, fill_target):
    #     top = pending_cfg[-1]
    #     for key, val in top.items():
    #         if val == None and key == "{}_pointer".format(fill_target):
    #             continue
    #         else:
    #             return False
    #     return True

    
    def build_functional(self):
        results = []
        cur_level = [self.root]
        cur_config = None
        while len(cur_level) > 0:
            next_level = []
            for node in cur_level:
                if not node.children:
                    continue
                next_level+=node.children
                if node.token == "<o>" and not node.key:
                    cur_config = self.attempt_fill_config(node.children)
                    results.append(cur_config)
            cur_level = next_level
        return results
    def attempt_fill_config(self, nodes):
        cur_config = dict(
                        type = "",
                        color = "",
                        action = "",
                        state = "",
                        pos = ""
                    )
        for node in nodes:
            if node.key:
                if node.token == "<dir>":
                    cur_config["pos"] = None
                elif node.token == "<s>":
                    cur_config["state"] = node.children[0].token
                elif node.token == "<p>":
                    cur_config["color"] = node.children[0].token
                elif node.token == "<t>":
                    cur_config["type"] = node.children[0].token
                elif node.token == "<a>":
                    cur_config["action"] = node.children[0].token
                else:
                    exit("Incorrect built tree built")
        return cur_config


                
            
    
    
    

        
                
        


    
              
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
            s = object_dict[obj_id]['<s>'] if object_dict[obj_id]['<s>'] != 'nil' else ''
            p = object_dict[obj_id]['<p>'] if object_dict[obj_id]['<p>'] != 'nil' else ''
            t = object_dict[obj_id]['<t>'] if object_dict[obj_id]['<t>'] != 'nil' else ''
            dir = ''
            if isinstance(object_dict[obj_id]['<dir>'], str):
                dir = object_dict[obj_id]['<a>'] if object_dict[obj_id]['<a>'] != 'nil' else ''
            elif isinstance(object_dict[obj_id]['<dir>'], dict):
                tdir = object_dict[obj_id]['<dir>']['<tdir>']
                tdir_mapping = {
                    "left": "on the left of",
                    "right": "on the right of",
                    "front": "in front of",
                    "back": "behind",
                    "left and front": "on the left and in front of",
                    "right and front": "on the right and in front of",
                    "left and back": "on the left and behind",
                    "right and back": "on the right and behind"
                }
                new_o = recur_translate(object_dict[obj_id]['<dir>']["<o>'s id"])
                dir = tdir_mapping[tdir] + ' ' + new_o
            else:
                print("warning!")
            a = ''
            if isinstance(object_dict[obj_id]['<a>'], str):
                a = object_dict[obj_id]['<a>'] if object_dict[obj_id]['<a>'] != 'nil' else ''
            elif isinstance(object_dict[obj_id]['<a>'], dict):
                if '<deed_with_o>' in object_dict[obj_id]['<a>'].keys():
                    deed_with_o = object_dict[obj_id]['<a>']['<deed_with_o>']
                    new_o = recur_translate(object_dict[obj_id]['<a>']["<o>'s id"])
                    a = deed_with_o + ' ' + new_o
                else:
                    a = object_dict[obj_id]['<a>']['<deed_without_o>']
            else:
                print("warning!")
            result = ""
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
            return result
    return recur_translate(0)


mytree = tree(4)
#print(mytree.depth)
print(mytree.root.visualize())
print(mytree.depth)
print(mytree.root.key)
print(mytree.build_functional())
better_visualize_tree(mytree.root)
print(mytree.new_build_functional())
print(translate(mytree.new_build_functional()))

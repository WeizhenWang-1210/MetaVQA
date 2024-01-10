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


                
            
    
    
    

        
                
        


    
              

        

mytree = tree(3)
#print(mytree.depth)
print(mytree.root.visualize())
print(mytree.depth)
print(mytree.root.key)
print(mytree.build_functional())


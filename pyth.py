from nodes import *
from data import *
import sys

class UnimplementedError(Exception):
    def __init__(self, active_char, rest_code):
        self.active_char = active_char
        self.rest_code = rest_code

    def __str__(self):
        return "%s is not implemented, %d from the end." % \
            (self.active_char, len(self.rest_code) + 1)

def construct_ast(code):
    ast = Root()
    parent = ast
    *rest_code, = code #Make code into list so we can use pop
    rest_code += [";"]
    active_char = ""
    inits = []

    while rest_code:
        active_char = rest_code.pop(0)
        if active_char == "." and rest_code[0] not in digits:
            active_char += rest_code.pop(0)

        #Rise up the tree if arity exhausted
        while len(parent.children) == parent.arity:
            parent = parent.parent
            #Check augmented assignment
            if isinstance(parent, (Assign, Post_Assign)) and not isinstance(parent.children[0], Variable):
                #DFS down tree for first variable
                nodes_left = parent.children[:]
                while nodes_left:
                    current_node = nodes_left.pop(0)
                    if isinstance(current_node, Variable):
                        parent.children.insert(0, copy.deepcopy(current_node))
                        break
                    nodes_left = current_node.children + nodes_left
                else:
                    raise ValueError("Assignment needs at least one Variable.")

        if active_char == ")":
            parent = parent.parent

        elif active_char == ";":
            while parent is not ast:
                if parent.arity is not UNBOUNDED and len(parent.children) < parent.arity:
                    parent.children += [Variable("Q") for _ in range(parent.arity - len(parent.children))]
                    if "Q" not in inits:
                        inits += ["Q"]
                parent = parent.parent

        #Parse numbers
        elif active_char in digits:
            number = active_char
            if  number != "0":
                while rest_code and rest_code[0] in digits and \
                    not (rest_code[0]=="." and "." in number):
                    number += rest_code.pop(0)
            parent = parent.append_child(Literal(eval(number)))

        elif active_char == '"':
            string = ""
            while rest_code:
                active_char = rest_code.pop(0)

                if active_char == "\\" and rest_code and rest_code[0] in '"\\':
                    active_char = rest_code.pop(0)
                elif active_char == '"':
                    break
                string += active_char

            parent = parent.append_child(Literal(string))

        #Char-Escape
        elif active_char == "\\":
            parent = parent.append_child(Literal(rest_code.pop(0)))

        #Check meta-operators from data.py
        elif active_char in meta_ops and not len(parent.children) and parent.arity > 0:
            operator = meta_ops[active_char](parent)
            if operator.arity == ARITY_VARIABLE or parent.arity <= operator.arity:
                operator.arity = parent.arity
            parent.parent.children.pop()
            parent.parent.append_child(operator)
            parent = operator

        #Parse operators from data.py
        elif active_char in operators:
            operator = operators[active_char]
            parent = parent.append_child(operator())

        elif active_char in Variable.env:
            if active_char not in inits:
                inits += [active_char]

                if active_char in "KJ":
                    rest_code = ["="] + [active_char] + rest_code
                    continue

            parent = parent.append_child(Variable(active_char))

        #That char is unimplemented, raise error
        else:
            raise UnimplementedError(active_char, rest_code)

    #Post-processing
    if "Q" in inits:
        ast.children[0:0] = [Init_Q()]

    if "z" in inits:
        ast.children[0:0] = [Init_z()]

    return ast

def run_ast(ast):
    ast.eval()

if __name__ == "__main__":
    code = sys.argv[-1]
    ast = construct_ast(code)

    if "-d" in sys.argv:
        print(code)
        print(ast)
        print()

    run_ast(ast)

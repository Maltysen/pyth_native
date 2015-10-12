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
    active_char = ""

    while rest_code:
        active_char = rest_code.pop(0)

        #Rise up the tree if arity exhausted
        while len(parent.children) == parent.arity:
            parent = parent.parent

        if active_char == ")":
            parent = parent.parent

        elif active_char ==";":
            parent = ast

        #Parse numbers
        elif active_char in digits:
            number = active_char
            while rest_code and rest_code[0] in digits and \
                not (rest_code[0]=="." and "." in number):
                number += rest_code.pop(0)
            parent = parent.append_child(Literal(eval(number)))

        #Char-Escape
        elif active_char == "\\":
            parent = parent.append_child(Literal(rest_code.pop(0)))

        #Check meta-operators from data.py
        elif active_char in meta_ops and not len(parent.children) and parent.arity > 0:
            operator = meta_ops[active_char](parent)
            parent.parent.children.pop()
            parent.parent.append_child(operator)
            parent = operator

        #Parse operators from data.py
        elif active_char in operators:
            operator = operators[active_char]
            parent = parent.append_child(operator())

        elif active_char in variables:
            parent = parent.append_child(Variable(active_char))

        #That char is unimplemented, raise error
        else:
            raise UnimplementedError(active_char, rest_code)

    return ast

def run_ast(ast):
    ast.eval()

if __name__ == "__main__":
    code = sys.argv[-1]
    ast = construct_ast(code)

    if "-d" in sys.argv:
        print(ast)
        print()

    run_ast(ast)

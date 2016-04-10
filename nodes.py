from copy import deepcopy
from functools import reduce
import numbers
import itertools
import collections

UNBOUNDED = ARITY_VARIABLE = -1

# Type checking
def is_num(a):
    return isinstance(a, numbers.Number)

def is_seq(a):
    return isinstance(a, collections.Sequence)


def is_col(a):
    return isinstance(a, collections.Iterable)


def is_hash(a):
    return isinstance(a, collections.Hashable)


def is_lst(a):
    return isinstance(a, list) or isinstance(a, tuple)

def itertools_norm(func, a, *args, **kwargs):
    if isinstance(a, str):
        return ["".join(group) for group in func(a, *args, **kwargs)]
    if isinstance(a, set):
        return [set(group) for group in func(a, *args, **kwargs)]
    else:
        return [list(group) for group in func(a, *args, **kwargs)]

class Node(object):
    def __init__(self):
        self.children = []

    def eval(self):
        return None

    def append_child(self, child):
        self.children.append(child)
        child.parent = self

        return child

    def first_child(self):
        return self.children[0]

    def last_child(self):
        return self.children[-1]

    def __str__(self):
        return self.__class__.__name__ + ": {" + ", ".join(map(str, self.children)) + "}"

class Statement(Node):
    arity = UNBOUNDED

    def append_child(self, child):
        if not isinstance(child, (Suppress_Imp_Print, Statement, Assign)) \
            and len(self.children) >= self.block_from:
            print_node = Imp_Print()
            print_node.append_child(child)

            super().append_child(print_node)

        else:
            super().append_child(child)

        return child

    def eval_block(self):
        for expression in self.children[self.block_from:]:
            expression.eval()

class Operator(Node):
    def eval(self):
        return self.operate(*[child.eval() for child in self.children])

    def operate(self):
        pass

class Control_Flow(Operator):
    def eval(self):
        return self.operate(*self.children)

class Lambda(Node):
    arity = 1

    def __init__(self, params):
        super().__init__()
        self.params = params

    def eval(self, *args):
        global variables
        old_vars = deepcopy(variables)

        for param, value in zip(self.params, args):
            variables[param] = value

        return_val = self.first_child().eval()

        variables = old_vars
        return return_val

class Lambda_Container(Control_Flow):
    def append_child(self, child):
        if not self.children:
            return super().append_child(Lambda(self.params)).append_child(child)
        return super().append_child(child)

class Root(Statement):
    block_from = 0

    def eval(self):
        self.eval_block()

    def __init__(self):
        super().__init__()
        self.parent = self

class Imp_Print(Operator):
    arity = 1

    def operate(self, a):
        return print(a)

class Literal(Node):
    arity = 0

    def __init__(self, value):
        super().__init__()
        self.value = value

    def eval(self):
        return self.value

    def __str__(self):
        return "Literal(" + repr(self.value) + ")"

class Suppress_Imp_Print(Operator):
    arity = 1

    def operate(self, a = None):
        return a if a != None else ()

class Negate(Operator):
    arity = 1

    def operate(self, a):
        return not a

class ErrorLoop(Control_Flow, Statement):
    block_from = 0

    def operate(self, *args):
        while True:
            try:
                self.eval_block()
            except:
                break

class Mod(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, int) and is_seq(b):
            return b[::a]
        if isinstance(a, complex) and is_num(b):
            return (a.real % b) + (a.imag % b) * 1j
        if is_num(a) and is_num(b):
            return a % b
        if isinstance(a, str):
            if is_lst(b):
                return a % tuple(b)
            else:
                return a % b
        raise BadTypeCombinationError("%", a, b)

class And(Control_Flow):
    arity = 2

    def operate(sefl, a, b):
        return a.eval() and b.eval()

class Tuple(Operator):
    arity = UNBOUNDED

    def operate(self, *args):
        return args

class Mul(Operator):
    arity = 2

    def operate(self, a, b):
        if is_col(a) and is_col(b):
            prod = list(itertools.product(a, b))
            if isinstance(a, str) and isinstance(b, str):
                return list(map(''.join, prod))
            else:
                return list(map(list, prod))
        if is_num(a) and is_num(b) or\
                isinstance(a, int) and is_seq(b) or\
                is_seq(a) and isinstance(b, int):
            return a * b
        raise BadTypeCombinationError("*", a, b)

class Plus(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, set):
            if is_col(b):
                return a.union(b)
            else:
                return a.union({b})
        if is_lst(a) and not is_lst(b):
            return list(a) + [b]
        if is_lst(b) and not is_lst(a):
            return [a] + list(b)
        if is_lst(a) and is_lst(b):
            return list(a) + list(b)
        if is_num(a) and is_num(b) or\
                isinstance(a, str) and isinstance(b, str):
            return a + b
        if is_num(a) and isinstance(b, str):
            return str(a) + b
        if isinstance(a, str) and is_num(b):
            return a + str(b)
        raise BadTypeCombinationError("+", a, b)

class Couple(Operator):
    arity = 2

    def operate(self, a, b):
        return [a, b]

class Minus(Operator):
    arity = 2

    def operate(self, a, b):
        if is_num(a) and is_num(b):
            return a - b
        if is_num(a) and is_col(b):
            if isinstance(b, str):
                return minus(str(a), b)
            if is_lst(b):
                return minus([a], b)
            if isinstance(b, set):
                return minus({a}, b)
        if is_num(b) and is_col(a):
            if isinstance(a, str):
                return minus(a, str(b))
            if is_lst(a):
                return minus(a, [b])
            if isinstance(a, set):
                return minus(a, {b})
        if is_col(a) and is_col(b):
            if isinstance(b, str):
                difference =\
                    filter(lambda c: not isinstance(c, str) or c not in b, a)
            else:
                difference = filter(lambda c: c not in b, a)
            if isinstance(a, str):
                return ''.join(difference)
            if is_lst(a):
                return list(difference)
            if isinstance(a, set):
                return set(difference)
        raise BadTypeCombinationError("-", a, b)
    environment['minus'] = minus

class Div(Operator):
    arity = 2

    def operate(self, a, b):
        if is_num(a) and is_num(b):
            return int(a // b)
        if is_seq(a):
            return a.count(b)
        raise BadTypeCombinationError("/", a, b)

variables = {
        "T": 10,
        "Z": 0,
        "Y": [],
        "d": " ",
        "N": '"',
        "G": "abcdefghijklmnopqrstuvwxyz"
}

class Variable(Node):
    arity = 0

    def __init__(self, name):
        super().__init__()
        self.name = name

    def eval(self):
        return variables[self.name]

    def __str__(self):
        return "Variable(" + self.name + ")"

class Assign(Control_Flow):
    arity = 2

    def operate(self, a, b):
        value = b.eval()

        variables[a.name] = value
        return value

class Post_Assign(Control_Flow):
    arity = 2

    def operate(self, a, b):
        old_value = a.eval()

        variables[a.name] = b.eval()
        return old_value

class If_Statement(Control_Flow, Statement):
    block_from = 1

    def operate(self, a, *args):
        self.true = False

        if a.eval():
            self.true = True
            self.eval_block()

class For_Loop(Control_Flow, Statement):
    block_from = 2

    def operate(self, a, b, *args):
        for value in b.eval():
            variables[a.name] = value
            self.eval_block()

class While_Loop(Control_Flow, Statement):
    block_from = 1

    def operate(self, a, *args):
        while a.eval():
            self.eval_block()

class Map(Lambda_Container):
    arity = 2
    params = ["d"]

    def operate(self, a, b):
        return [a.eval(value) for value in b.eval()]

class First_N(Lambda_Container):
    arity = 3
    params = ["Z"]

    def operate(a, b, c=1):
        return list(itertools.islice(filter(a, infinite_iterator(c)), b))

# class Unary_Map(Meta_Param):
#      arity = 1
#
#      def operate(self, a):
#          return [self.op.eval_with_children(i) for i in a]

class Neg(Operator):
    arity = 1

    def operate(self, a):
        return -a

class Float_Div(Operator):
    arity = 2

    def operate(self, a, b):
        return a / b

class List(Operator):
    arity = UNBOUNDED

    def operate(self, *args):
        return list(args)

class One_List(Operator):
    arity = 1

    def operate(self, a):
        return [a]

class Ternary(Control_Flow):
    arity = 3

    def operate(self, a, b, c):
        if a.eval():
            return b.eval()
        return c.eval()

class Sum(Operator):
    arity = 1

    def operate(self, a):
        return sum(a)

class Range(Operator):
    arity = 2

    def operate(self, a, b):
        return list(range(a, b))

class Urange(Operator):
    arity = 1

    def operate(self, a):
        return list(range(a))

class Equals(Operator):
    arity = 2

    def operate(self, a, b):
        return a == b

class Head(Operator):
    arity = 1

    def operate(self, a):
        return a + 1

class Repr(Operator):
    arity = 1

    def operate(self, a):
        return repr(a)

class Char(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int):
            return chr(a)
        if is_num(a):
            return a.real - a.imag * 1j
        if isinstance(a, str):
            return to_base_ten(list(map(ord, a)), 256)
        if is_col(a):
            trans = list(zip(*a))
            if all(isinstance(sublist, str) for sublist in a):
                return list(map(''.join, trans))
            else:
                return list(map(list, trans))
        raise BadTypeCombinationError("C", a)

class Primes_Pop(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int):
            if a < 2:
                return []
            try:
                from sympy import factorint
                factor_dict = factorint(a)
                factors_with_mult = [[fact for _ in range(factor_dict[fact])] for fact in factor_dict]
                return sorted(sum(factors_with_mult, []))
            except:
                working = a
                output = []
                num = 2
                while num * num <= working:
                    while working % num == 0:
                        output.append(num)
                        working //= num
                    num += 1
                if working != 1:
                    output.append(working)
                return output
        if is_num(a):
            return cmath.phase(a)
        if is_seq(a):
            return a[:-1]
        raise BadTypeCombinationError("P", a)

def to_base_ten(arb, base):
    # Special cases
    if abs(base) == 1:
        return len(arb)
    acc = 0
    for digit in arb:
        acc *= base
        acc += digit
    return acc

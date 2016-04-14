import copy
from functools import reduce
import numbers
import itertools
import ast
import math
import cmath
import string
import random
import binascii
import collections

UNBOUNDED = ARITY_VARIABLE = -1

# Error handling
class BadTypeCombinationError(Exception):

    def __init__(self, func, *args):
        self.args = args
        self.func = func

    def __str__(self):
        error_message = "\nError occured in function: %s" % self.func
        for i in range(len(self.args)):
            arg = self.args[i]
            arg_type = str(type(arg)).split("'")[1]
            error_message += "\nArg %d: %r, type %s." % (i + 1, arg, arg_type)
        return error_message

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

def to_base_ten(arb, base):
    # Special cases
    if abs(base) == 1:
        return len(arb)
    acc = 0
    for digit in arb:
        acc *= base
        acc += digit
    return acc

def from_base_ten(arb, base):
    # Special cases
    if arb == 0:
        return [0]
    if abs(base) == 1:
        return [0] * arb
    # Main routine
    base_list = []
    work = arb
    while work != 0:
        work, remainder = divmod(work, base)
        if remainder < 0:
            work += 1
            remainder -= base
        if work == -1 and base > 0:
            work = 0
            remainder -= base
        base_list.append(remainder)
    return base_list[::-1]

def urange(a):
    if isinstance(a, int):
        if a >= 0:
            return list(range(a))
        else:
            return list(range(a, 0))
    if is_num(a):
        return urange(int(a))
    if is_col(a):
        return list(range(len(a)))
    raise BadTypeCombinationError("U", a)

def num_to_range(arg):
    if is_num(arg):
        return urange(int(arg))

    return arg

def equal(a, b=None):
    if b is None:
        if is_seq(a):
            if not a:
                return True
            return all(a[0] == a_elem for a_elem in a)
    return a == b

def hex_multitype(a, func):
    if isinstance(a, str):
        return "0x" + (binascii.hexlify(a.encode("utf-8")).decode("utf-8") or "0")

    if isinstance(a, int):
        return hex(a)

    raise BadTypeCombinationError(func, a)

def preprocess_eval(a):
    if isinstance(a, str):
        if a and a[0] == '0':
            to_eval = a.lstrip('0')
            if not to_eval or not to_eval[0].isdecimal():
                to_eval = '0' + to_eval
            return to_eval
        else:
            return a
    raise BadTypeCombinationError('v', a)

def Pliteral_eval(a):
    if isinstance(a, str):
        return ast.literal_eval(preprocess_eval(a))
    raise BadTypeCombinationError('v', a)


def Punsafe_eval(a):
    if isinstance(a, str):
        return eval(preprocess_eval(a))
    raise BadTypeCombinationError('v', a)

python_eval = Punsafe_eval

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
        if not isinstance(child, (Suppress_Imp_Print, Statement, Assign, Explicit_Print)) \
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
        old_vars = copy.deepcopy(Variable.env)

        for param, value in zip(self.params, args):
            Variable.env[param] = value

        return_val = self.first_child().eval()

        Variable.env = old_vars
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
        print(a, flush=True)
        return a


class Literal(Node):
    arity = 0

    def __init__(self, value):
        super().__init__()
        self.value = value

    def eval(self):
        return self.value

    def __str__(self):
        return "Literal(" + repr(self.value) + ")"

class Variable(Node):
    arity = 0

    def __init__(self, name):
        super().__init__()
        self.name = name

    def eval(self):
        return Variable.env[self.name]

    def __str__(self):
        return "Variable(" + self.name + ")"

class Explicit_Print(Operator):
    arity = 1

    def operate(self, a):
        print(a, flush=True)
        return a

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
                return self.operate(str(a), b)
            if is_lst(b):
                return self.operate([a], b)
            if isinstance(b, set):
                return self.operate({a}, b)
        if is_num(b) and is_col(a):
            if isinstance(a, str):
                return self.operate(a, str(b))
            if is_lst(a):
                return self.operate(a, [b])
            if isinstance(a, set):
                return self.operate(a, {b})
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

class Div(Operator):
    arity = 2

    def operate(self, a, b):
        if is_num(a) and is_num(b):
            return int(a // b)
        if is_seq(a):
            return a.count(b)
        raise BadTypeCombinationError("/", a, b)

class Slice(Operator):
    arity = 3

    def operate(self, a, b, c=0):
        if isinstance(a, str) and isinstance(b, str):
            if isinstance(c, str):
                return re.sub(b, c, a)
            if c == 0:
                return bool(re.search(b, a))
            if c == 1:
                return [m.group(0) for m in re.finditer(b, a)]
            if c == 2:
                def first_group(matchobj):
                    return matchobj.group(1)
                return re.sub(b, first_group, a)
            if c == 3:
                return re.split(b, a)
            if c == 4:
                return [[m.group(0)] + list(m.groups()) for m in re.finditer(b, a)]
            raise BadTypeCombinationError(":", a, b, c)
        if is_seq(a) and isinstance(b, int) and isinstance(c, int):
            return a[slice(b, c)]

        if is_num(a) and is_num(b) and is_num(c):
            if c > 0:
                work = a
                gen_range = []
                if a <= b:
                    def cont_test(work): return work < b
                    step = c
                else:
                    def cont_test(work): return work > b
                    step = -c
                while cont_test(work):
                    gen_range.append(work)
                    work += step
                return gen_range
            elif c < 0:
                return at_slice(b, a, -c)[::-1]

        # There is no nice ABC for this check.
        if hasattr(a, "__getitem__") and is_col(b):
            if is_col(c):
                c = itertools.cycle(c)
            else:
                c = itertools.repeat(c)

            if isinstance(a, str) or isinstance(a, tuple):
                indexable = list(a)
            else:
                indexable = a

            for index in b:
                if isinstance(a, str):
                    indexable[index] = str(next(c))
                else:
                    indexable[index] = next(c)

            if isinstance(a, str):
                return "".join(indexable)

            return indexable

        raise BadTypeCombinationError(":", a, b, c)

class Less(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, set) and is_col(b):
            return a.issubset(b) and a != b
        if is_seq(a) and is_num(b):
            return a[:b]
        if is_num(a) and is_seq(b):
            return b[:len(b)-a]
        if isinstance(a, complex) or isinstance(b, complex):
            return abs(a) < abs(b)
        if is_num(a) and is_num(b) or\
                isinstance(a, list) and isinstance(b, list) or\
                isinstance(a, tuple) and isinstance(b, tuple) or\
                isinstance(a, str) and isinstance(b, str):
            return a < b
        raise BadTypeCombinationError("<", a, b)

class Assign(Control_Flow): #TODO: Make augmented work
    arity = 2

    def operate(self, a, b):
        value = b.eval()

        Variable.env[a.name] = value
        return value

class Greater(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, set) and is_col(b):
            return a.issuperset(b) and a != b
        if is_seq(a) and is_num(b):
            return a[b:]
        if is_num(a) and is_seq(b):
            return b[len(b)-a:]
        if isinstance(a, complex) or isinstance(b, complex):
            return abs(a) > abs(b)
        if is_num(a) and is_num(b) or\
                isinstance(a, list) and isinstance(b, list) or\
                isinstance(a, tuple) and isinstance(b, tuple) or\
                isinstance(a, str) and isinstance(b, str):
            return a > b
        raise BadTypeCombinationError(">", a, b)

class Ternary(Control_Flow):
    arity = 3

    def operate(self, a, b, c):
        if a.eval():
            return b.eval()
        return c.eval()

class Lookup(Operator):
    arity = 2

    def operate(self, a, b):
        if is_num(a) and is_num(b):
            return a ** (1 / b)
        if isinstance(a, dict):
            if isinstance(b, list):
                b = tuple(b)
            return a[b]
        if is_seq(a) and isinstance(b, int):
            return a[b % len(a)]
        if is_col(a) and is_col(b):
            if isinstance(a, str):
                intersection = filter(lambda b_elm: isinstance(b_elm, str)
                                      and b_elm in a, b)
            else:
                intersection = filter(lambda b_elem: b_elem in a, b)
            if isinstance(a, str):
                return ''.join(intersection)
            if isinstance(a, set):
                return set(intersection)
            else:
                return list(intersection)
        raise BadTypeCombinationError("@", a, b)

class Double_Assign(Operator):
    arity = 1

    def operate(self, a):
        if len(a) == 2:
            Variable.env["G"], Variable.env["H"] = a
            return a

        raise BadTypeCombinationError("A", a)

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

class Eval_Input(Operator):
    arity = 0

    def operate(self):
        return python_eval(input())

class For_Loop(Control_Flow, Statement):
    block_from = 2

    def operate(self, a, b, *args):
        for value in num_to_range(b.eval()):
            Variable.env[a.name] = value
            self.eval_block()

class If_Statement(Control_Flow, Statement):
    block_from = 1

    def operate(self, a, *args):
        self.true = False

        if a.eval():
            self.true = True
            self.eval_block()

class Random(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int):
            if a == 0:
                return random.random()
            if a < 0:
                random.seed(-a)
                return
            if a > 0:
                return random.choice(urange(a))
        if is_col(a):
            return random.choice(list(a))
        raise BadTypeCombinationError("O", a)

class Primes_Pop(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int):
            if a < 0:
                # Primality testing
                return len(self.operate(-a)) == 1
            if a < 2:
                return []
            try:
                from sympy import factorint
                factor_dict = factorint(a)
                factors_with_mult = [[fact for _ in range(
                    factor_dict[fact])] for fact in factor_dict]
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

class Init_Q(Node):
    arity = 0

    def eval(self):
        Variable.env["Q"] = python_eval(input())

    def __str__(self):
        return "Init_Q()"

class Init_z(Node):
    arity = 0

    def eval(self):
        Variable.env["z"] = input()

    def __str__(self):
        return "Init_z()"

class Sort(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, str):
            return ''.join(sorted(a))
        if is_col(a):
            return sorted(a)
        if isinstance(a, int):
            return list(range(1, a + 1))
        if is_num(a):
            return Psorted(int(a))
        raise BadTypeCombinationError("S", a)

class Urange(Operator):
    arity = 1

    def operate(self, a):
        return urange(a)

class Implicit_For(Control_Flow, Statement):
    block_from = 1

    def operate(self, a, *args):
        for value in num_to_range(a.eval()):
            Variable.env["N"] = value
            self.eval_block()

class While_Loop(Control_Flow, Statement):
    block_from = 1

    def operate(self, a, *args):
        while a.eval():
            self.eval_block()

class Assign_At(Operator):
    arity = 3

    def operate(self, a, b, c):
        # Assign at
        if isinstance(a, dict):
            if isinstance(b, list):
                b = tuple(b)
            a[b] = c
            return a
        if isinstance(b, int):
            if isinstance(a, list):
                a[b % len(a)] = c
                return a
            if isinstance(a, str):
                return a[:b % len(a)] + str(c) + a[(b % len(a)) + 1:]
            if isinstance(a, tuple):
                return a[:b % len(a)] + (c,) + a[(b % len(a)) + 1:]
            raise BadTypeCombinationError("X", a, b, c)
        # Translate
        if is_seq(a) and is_seq(b) and (c is None or is_seq(c)):
            if c is None:
                c = b[::-1]

            def trans_func(element):
                return c[b.index(element) % len(c)] if element in b else element
            translation = map(trans_func, a)
            if isinstance(a, str) and isinstance(c, str):
                return ''.join(translation)
            else:
                return list(translation)
        # += in a list, X<int><list><any>
        if isinstance(a, int) and is_lst(b):
            b[a % len(b)] = plus(b[a % len(b)], c)
            return b
        # += in a dict, X<any><dict><any>
        if isinstance(b, dict):
            if isinstance(a, list):
                a = tuple(a)
            if a in b:
                b[a] = plus(b[a], c)
            else:
                b[a] = c
            return b
        # Insert in a string, X<int><str><any>
        if isinstance(a, int) and isinstance(b, str):
            if not isinstance(c, str):
                c = str(c)
            return b[:a] + c + b[a:]
        raise BadTypeCombinationError("X", a, b, c)


class List(Operator):
    arity = UNBOUNDED

    def operate(self, *args):
        return list(args)

class One_List(Operator):
    arity = 1

    def operate(self, a):
        return [a]

class Exponentiate(Operator):
    arity = 2

    def operate(self, a, b):
        if is_num(a) and is_num(b):
            return pow(a, b)
        if is_col(a) and isinstance(b, int):
            return itertools_norm(itertools.product, a, repeat=b)

        raise BadTypeCombinationError("^", a, b)

class Opposite(Operator):
    arity = 1

    def operate(self, a):
        if is_num(a):
            return -a
        if is_seq(a):
            return a[::-1]
        if isinstance(a, dict):
            return {value: key for key, value in a.items()}
        raise BadTypeCombinationError("_", a)

class Repr(Operator):
    arity = 1

    def operate(self, a):
        return repr(a)

class Append(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, list):
            a.append(b)
            return a
        if isinstance(a, set):
            if is_hash(b):
                a.add(b)
                return a
            else:
                a.add(tuple(b))
                return a
        if is_num(a) and is_num(b):
            return abs(a - b)
        raise BadTypeCombinationError("a", a, b)

class Chop(Operator):
    arity = 2

    def operate(self, a, b=None):
        if is_num(a) and is_num(b):
            return a / b
        if isinstance(a, str) and isinstance(b, str):
            return a.split(b)
        if isinstance(a, str) and b is None:
            return a.split()
        # iterable, int -> chop a into pieces of length b
        if is_seq(a) and isinstance(b, int):
            return [a[i:i + b] for i in range(0, len(a), b)]
        # int, iterable -> split b into a pieces (distributed equally)
        if isinstance(a, int) and is_seq(b):
            m = len(b) // a  # min number of elements
            r = len(b) % a   # remainding elements
            begin, end = 0, m + (r > 0)
            l = []
            for i in range(a):
                l.append(b[begin:end])
                begin, end = end, end + m + (i + 1 < r)
            return l
        # seq, col of ints -> chop seq at number locations.
        if is_seq(a) and is_col(b):
            if all(isinstance(elem, int) for elem in b) and not isinstance(b, str):
                locs = sorted(b)
                return list(map(lambda i, j: a[i:j], [0] + locs, locs + [len(a)]))
        if is_seq(a):
            output = [[]]
            for elem in a:
                if elem == b:
                    output.append([])
                else:
                    output[-1].append(elem)
            return output
        raise BadTypeCombinationError("c", a, b)

class End(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, complex):
            return a.imag
        if is_num(a):
            return a % 10
        if is_seq(a):
            return a[-1]
        raise BadTypeCombinationError("e", a)

class Pfilter(Lambda_Container):
    arity = 2
    params = ["T"]

    def operate(self, a, b=None):
        b = b.eval() if b != None else 1

        if is_num(b):
            return next(filter(a.eval, itertools.count(b)))
        if is_col(b):
            return list(filter(a.eval, b))
        raise BadTypeCombinationError("f", a, b)

class Greater_Equal(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, set) and is_col(b):
            return a.issuperset(b)
        if is_seq(a) and is_num(b):
            return a[b - 1:]
        if is_num(a) and is_num(b) or\
                isinstance(a, list) and isinstance(b, list) or\
                isinstance(a, tuple) and isinstance(b, tuple) or\
                isinstance(a, str) and isinstance(b, str):
            return a >= b
        raise BadTypeCombinationError("g", a, b)

class Head(Operator):
    arity = 1

    def operate(self, a):
        if is_num(a):
            return a + 1
        if is_seq(a):
            return a[0]
        raise BadTypeCombinationError("h", a)

class Base10(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, str) and isinstance(b, int):
            if not a:
                return 0
            return int(a, b)
        if is_seq(a) and is_num(b):
            return to_base_ten(a, b)
        if isinstance(a, int) and isinstance(b, int):
            return fractions.gcd(a, b)
        raise BadTypeCombinationError("i", a, b)

class Join(Operator):
    arity = 2

    def operate(self, a, b):
        if b is None:
            a, b = '\n', a
        if isinstance(a, int) and isinstance(b, int):
            return from_base_ten(a, b)
        if isinstance(a, str) and is_col(b):
            return a.join(list(map(str, b)))
        if is_col(b):
            return str(a).join(list(map(str, b)))
        raise BadTypeCombinationError("j", a, b)

class Length(Operator):
    arity = 1

    def operate(self, a):
        if is_num(a):
            if isinstance(a, complex) or a < 0:
                return cmath.log(a, 2)
            return math.log(a, 2)

        if is_col(a):
            return len(a)
        raise BadTypeCombinationError("l", a)

class Map(Lambda_Container):
    arity = 2
    params = ["d"]

    def operate(self, a, b):
        b = b.eval()

        if is_num(b):
            return [a.eval(value) for value in urange(b)]
        if is_col(b):
            return [a.eval(value) for value in b]
        raise BadTypeCombinationError("m", a, b)

class Not_Equal(Operator):
    arity = 2

    def operate(self, a, b=None):
        return not equal(a, b)

class Order_By(Lambda_Container):
    arity = 2
    params = ["N"]

    def operate(self, a, b):
        b = b.eval()

        if is_num(b):
            b = urange(b)
        if is_col(b):
            if isinstance(b, str):
                return ''.join(sorted(b, key=a.eval))
            else:
                return sorted(b, key=a.eval)
        raise BadTypeCombinationError("o", a, b)

class Debug_Print(Operator):
    arity = 1

    def operate(self, a):
        print(a, flush=True, end="")
        return a

class Equals(Operator):
    arity = 2

    def operate(self, a, b=None):
        if b is None:
            if is_seq(a):
                if not a:
                    return True
                return all(a[0] == a_elem for a_elem in a)
        return a == b

class Range(Operator):
    arity = 2

    def operate(self, a, b):
        def run_length_encode(a):
            return [[len(list(group)), key] for key, group in itertools.groupby(a)]

        if isinstance(b, int):
            if isinstance(a, str):
                if b == 0:
                    return a.lower()
                if b == 1:
                    return a.upper()
                if b == 2:
                    return a.swapcase()
                if b == 3:
                    return a.title()
                if b == 4:
                    return a.capitalize()
                if b == 5:
                    return string.capwords(a)
                if b == 6:
                    return a.strip()
                if b == 7:
                    return [Pliteral_eval(part) for part in a.split()]
                if b == 8:
                    return run_length_encode(a)
                if b == 9:
                    # Run length decoding, format "<num><char><num><char>",
                    # e.g. "12W3N6S1E"
                    return re.sub(r'(\d+)(\D)',
                                  lambda match: int(match.group(1))
                                  * match.group(2), a)

            if is_seq(a):
                if b == 8:
                    return run_length_encode(a)
                if b == 9:
                    if all(isinstance(key, str) for group_size, key in a):
                        return ''.join(key * group_size for group_size, key in a)
                    else:
                        return sum(([copy.deepcopy(key)] * group_size
                                    for group_size, key in a), [])
                raise BadTypeCombinationError("r", a, b)

            if isinstance(a, int):
                if a < b:
                    return list(range(a, b))
                else:
                    return list(range(a, b, -1))
            raise BadTypeCombinationError("r", a, b)
        if isinstance(a, str) and isinstance(b, str):
            a_val = Pchr(a)
            b_val = Pchr(b)
            ab_range = Prange(a_val, b_val)
            return [''.join(chr(char_val) for char_val in join(str_val, 256))
                    for str_val in ab_range]

class Sum(Operator):
    arity = 1

    def operate(self, a):
        if is_col(a) and not isinstance(a, str):
            if len(a) == 0:
                return 0
            if all(isinstance(elem, str) for elem in a):
                return ''.join(a)
            if len(a) > 100:
                cutoff = len(a) // 2
                first = a[:cutoff]
                second = a[cutoff:]
                return plus(self.operate(first), self.operate(second))
            return reduce(lambda b, c: plus(b, c), a[1:], a[0])
        if isinstance(a, complex):
            return a.real
        if a == '':
            return 0
        if is_num(a) or isinstance(a, str):
            return int(a)
        raise BadTypeCombinationError("s", a)

class Tail(Operator):
    arity = 1

    def operate(self, a):
        if is_num(a):
            return a - 1
        if is_seq(a):
            return a[1:]
        raise BadTypeCombinationError("t", a)

class Reduce(Lambda_Container):
    arity = 3
    params = ["G", "H"]

    def operate(self, a, b, c=None):
        b = b.eval()
        c = c.eval() if c!=None else None

        # Fixed point / Loop
        if c is None:
            counter = 0
            results = [copy.deepcopy(b)]
            acc = a.eval(b, counter)
            while acc not in results:
                counter += 1
                results.append(copy.deepcopy(acc))
                acc = a.eval(acc, counter)
            return results[-1]

        # Reduce
        if is_seq(b) or is_num(b):
            if is_num(b):
                seq = urange(b)
            else:
                seq = b
            acc = c
            while len(seq) > 0:
                h = seq[0]
                acc = a.eval(acc, h)
                seq = seq[1:]
            return acc
        raise BadTypeCombinationError("u", a, b, c)

class Eval(Operator):
    arity = 1

    def operate(self, a):
        return python_eval(a)

class Raw_Input(Operator):
    arity = 0

    def operate(self):
        return input()

class Index(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a ^ b
        if is_seq(a) and not (isinstance(a, str) and not isinstance(b, str)):
            if b in a:
                return a.index(b)
            # replicate functionality from str.find
            else:
                return -1
        if is_lst(b):
            return [index for index, elem in enumerate(b) if elem == a]
        if isinstance(a, str):
            return index(a, str(b))
        raise BadTypeCombinationError("x", a, b)

class Powerset(Operator):
    arity = 1

    def operate(self, a):
        if is_num(a):
            return a * 2
        if is_col(a):
            def powerset(col):
                return itertools.chain.from_iterable(
                    itertools.combinations(col, i) for i in range(0, len(col) + 1))
            return itertools_norm(powerset, a)
        raise BadTypeCombinationError("y", a)

class Uniquify(Operator):
    arity = 1

    def operate(self, a):
        if is_seq(a):
            try:
                seen = set()
                out = []
                for elem in a:
                    if not elem in seen:
                        out.append(elem)
                        seen.add(elem)
            except:
                out = []
                for elem in a:
                    if not elem in out:
                        out.append(elem)
            if isinstance(a, str):
                return ''.join(out)
            return out
        if is_col(a):
            return sorted(a)
        raise BadTypeCombinationError('{', a)

class Or(Control_Flow):
    arity = 2

    def operate(self, a, b):
        return a.eval() or b.eval()

class In(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            if a < b:
                return list(range(a, b + 1))
            return list(range(b, a + 1))[::-1]
        return a in b

class Post_Assign(Control_Flow):
    arity = 2

    def operate(self, a, b):
        old_value = a.eval()

        Variable.env[a.name] = b.eval()
        return old_value

class Factorial(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int):
            return math.factorial(a)
        if is_num(a):
            return math.gamma(a + 1)
        raise BadTypeCombinationError('.!', a)

class Hex(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int) or isinstance(a, str):
            return hex_multitype(a, ".H")[2:]
        raise BadTypeCombinationError(".H", a)

############################################################
# class First_N(Lambda_Container):
#     arity = 3
#     params = ["Z"]
#
#     def operate(a, b, c=1):
#         return list(itertools.islice(filter(a, infinite_iterator(c)), b))
#
# class Unary_Map(Meta_Param):
#      arity = 1
#
#      def operate(self, a):
#          return [self.op.eval_with_children(i) for i in a]

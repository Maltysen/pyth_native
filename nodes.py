import copy
from functools import reduce
import functools
import numbers
import itertools
import ast
import math
import cmath
import string
import random
import binascii
import collections
import zlib
import time
import sys

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

def infinite_iterator(start):
    def successor(char):
        if char.isalpha():
            if char == 'z':
                return 'a', True
            if char == 'Z':
                return 'A', True
            return chr(ord(char) + 1), False
        elif char.isdigit():
            if char == '9':
                return '0', True
            return chr(ord(char) + 1), False
        else:
            return chr(ord(char) + 1), False

    if is_num(start):
        while True:
            yield start
            start += 1

    # Replicates the behavior of ruby's .succ
    if isinstance(start, str):
        while True:
            yield start
            alphanum_locs = list(filter(lambda loc: start[loc].isalnum()
                                        and ord(start[loc]) < 128,
                                        range(len(start))))
            if alphanum_locs:
                locs = alphanum_locs[::-1]
            elif start:
                locs = range(len(start))[::-1]
            else:
                locs = []
                succ_char = 'a'
            for inc_loc in locs:
                inc_char = start[inc_loc]
                succ_char, carry = successor(inc_char)
                start = start[:inc_loc] + succ_char + start[inc_loc + 1:]
                if not carry:
                    break
            else:
                start = succ_char + start

    raise BadTypeCombinationError("infinite_iterator, probably .V", start)

@functools.lru_cache(1)
def all_input():
    return [l.rstrip("\n") for l in sys.stdin]

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

def literal_eval(a):
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

Lambda.__call__ = Lambda.eval

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
        if a is not None:
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

class Read_File(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, str):
            if any(a.lower().endswith("." + i) for i in
                   ["png", "jpg", "jpeg", "gif", "svg", "ppm", "pgm", "pbm"]):
                from PIL import Image
                img = Image.open(a)
                data = list(img.getdata())

                # If alpha all 255, take out alpha
                if len(data[0]) > 3 and all(i[3] == 255 for i in data):
                    data = [i[:3] for i in data]

                # Check grayscale
                if all(i.count(i[0]) == len(i) for i in data):
                    data = [i[0] for i in data]

                data = chop(data, img.size[0])
                return data

            if a.startswith("http"):
                b = urllib.request.urlopen(a)
            else:
                b = open(a)

            b = [lin[:-1] if lin[-1] == '\n' else lin for lin in b]
            return b

        raise BadTypeCombinationError("'", a)

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

class Func_Def(Lambda_Container):
    arity = 1
    params = ["b"]

    def operate(self, a):
        Powerset.operate = a.eval

class Func_Def2(Lambda_Container):
    arity = 1
    params = ["G", "H"]

    def operate(self, a):
        Greater_Equal.operate = a.eval

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

class Filter(Lambda_Container):
    arity = 2
    params = ["T"]

    def operate(self, a, b=None):
        b = b.eval() if b != None else 1

        if is_num(b):
            return next(filter(a, itertools.count(b)))
        if is_col(b):
            return list(filter(a, b))
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
            return [a(value) for value in urange(b)]
        if is_col(b):
            return [a(value) for value in b]
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
                return ''.join(sorted(b, key=a))
            else:
                return sorted(b, key=a)
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
            return reduce(lambda b, c: Plus.operate(None, b, c), a[1:], a[0])
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
            acc = a(b, counter)
            while acc not in results:
                counter += 1
                results.append(copy.deepcopy(acc))
                acc = a(acc, counter)
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
                acc = a(acc, h)
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

class Bitwise_And(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a & b

        raise BadTypeCombinationError(".&", a, b)

class Pop_Loc(Operator):
    arity = 2

    def operate(self, a, b):
        if is_lst(a) and isinstance(b, int):
            return a.pop(b)

        raise BadTypeCombinationError(".(", a, b)

class Pop(Operator):
    arity = 1

    def operate(self, a):
        if is_lst(a):
            return a.pop()

        raise BadTypeCombinationError(".)", a, b)

class Remove(Operator):
    arity = 2

    def operate(self, a, b):
        if not is_col(a) or not is_col(b):
            raise BadTypeCombinationError(".-", a, b)

        seq = list(a)
        to_remove = list(b)
        for elem in to_remove:
            if elem in seq:
                del seq[seq.index(elem)]

        if isinstance(a, str):
            return ''.join(seq)
        return seq

class Partition(Operator):
    arity = 1

    def operate(self, a):
        if is_seq(a):
            all_splits = []
            for n in range(len(a)):  # 0, 1, ..., len(a)-1 splits
                for idxs in itertools.combinations(range(1, len(a)), n):
                    all_splits.append(
                        [a[i:j] for i, j in zip((0,) + idxs, idxs + (None,))])
            return all_splits

        if isinstance(a, int) and a >= 0:
            @memoized
            def integer_partition(number):
                result = set()
                result.add((number, ))
                for x in range(1, number):
                    for y in integer_partition(number - x):
                        result.add(tuple(sorted((x, ) + y)))
                return result
            return list(map(list, sorted(integer_partition(a))))

        raise BadTypeCombinationError("./", a)

class Substrings(Operator):
    arity = 2

    def operate(self, a, b):
        if is_seq(a):
            seq = a
        elif is_num(a):
            seq = urange(a)
        else:
            raise BadTypeCombinationError(".:", a, b)
        if is_col(b):
            return sum(([seq[start:start + step]
                         for step in b if start + step <= len(seq)]
                        for start in range(len(seq))), [])
        if isinstance(b, int):
            step = b
        elif isinstance(b, float):
            step = int(b * len(seq))
        elif not b:
            all_substrs = [substrings(seq, step)
                           for step in range(1, len(seq) + 1)]
            return list(itertools.chain.from_iterable(all_substrs))
        else:
            raise BadTypeCombinationError(".:", a, b)
        return [seq[start:start + step] for start in range(len(seq) - step + 1)]

class Left_Shift(Operator):
    arity = 2

    def operate(self, a, b):
        if not isinstance(b, int):
            raise BadTypeCombinationError(".<", a, b)

        if is_seq(a):
            b %= len(a)
            return a[b:] + a[:b]

        if isinstance(a, int):
            return a << b

        raise BadTypeCombinationError(".<", a, b)

class Right_Shift(Operator):
    arity = 2

    def operate(self, a, b):
        if not isinstance(b, int):
            raise BadTypeCombinationError(".>", a, b)

        if is_seq(a):
            b %= len(a)
            return a[-b:] + a[:-b]

        if isinstance(a, int):
            return a >> b

        raise BadTypeCombinationError(".>", a, b)

class All(Operator):
    arity = 1

    def operate(self, a):
        if is_col(a):
            return all(a)

        raise BadTypeCombinationError(".A", a)

class Bin_Str(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int) or isinstance(a, str):
            return bin(int(hex_multitype(a, ".B"), 16))[2:]
        raise BadTypeCombinationError(".B", a)

class Combs_Repl(Operator):
    arity = 2

    def operate(self, a, b):
        if not is_col(a) or not isinstance(b, int):
            raise BadTypeCombinationError(".C", a, b)

        return itertools_norm(itertools.combinations_with_replacement, a, b)

class Divmod(Operator):
    arity = 2

    def operate(self, a, b):
        if is_num(a) and is_num(b):
            return list(divmod(a, b))
        elif is_seq(a) and is_num(b):
            return divmod_or_delete(a, [b])
        elif is_seq(a) and is_col(b):
            output = [e for i, e in enumerate(a) if i not in b]
            if isinstance(a, str):
                return "".join(output)
            return output
        else:
            raise BadTypeCombinationError('.D', a, b)

class Any(Operator):
    arity = 1

    def operate(self, a):
        if is_col(a):
            return any(a)
        if is_num(a):
            return int(math.ceil(a))
        raise BadTypeCombinationError(".E", a)

class Format_Str(Operator):
    arity = 2

    def operate(self, a, b):
        if not isinstance(a, str):
            raise BadTypeCombinationError(".F", a, b)
        if is_seq(b) and not isinstance(b, str):
            return a.format(*b)

        return a.format(b)

class Hex_Str(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int) or isinstance(a, str):
            return hex_multitype(a, ".H")[2:]
        raise BadTypeCombinationError(".H", a)

class Invert(Lambda_Container):
    arity = 2
    params = ["G"]

    def operate(self, a, b):
        b = b.eval()
        if not is_num(b):
            raise BadTypeCombinationError(".I", a, b)
        inv = 1.
        if a(inv) == b:
            return inv
        while a(inv) < b:
            inv *= 2
        delta = inv / 2
        while delta > 1e-20:
            if a(inv) == b:
                return inv
            if a(inv - delta) > b:
                inv -= delta
            elif a(inv - delta) == b:
                return inv - delta
            delta /= 2
        return inv

class Maximize(Lambda_Container):
    arity = 2
    params = ["Z"]

    def operate(self, a, b):
        b = b.eval()
        if is_num(b):
            seq = urange(b)
        elif is_col(b):
            seq = b
        else:
            raise BadTypeCombinationError(".M", a, b)
        maximum = max(map(a, seq))
        return list(filter(lambda elem: a(elem) == maximum, seq))

class Oct_Str(Operator):
    arity = 1

    def operate(self, a):
        if is_col(a) and all(map(is_num, a)):
            if len(a) == 0:
                return 0.0
            else:
                return sum(a) / len(a)
        elif isinstance(a, int) or isinstance(a, str):
            return oct(int(hex_multitype(a, ".O"), 16))[2:]
        raise BadTypeCombinationError(".O", a)

class Permutations2(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            # compute n P r
            return functools.reduce(operator.mul, range(a - b + 1, a + 1), 1)

        if is_col(a) and isinstance(b, int):
            return itertools_norm(itertools.permutations, a, b)

        if isinstance(a, int) and isinstance(b, str):
            return "".join(permutations2(a, list(b)))

        if isinstance(a, int) and is_col(b):
            # Algorithm modified from
            # http://stackoverflow.com/a/6784359/1938435
            # cc by-sa 3.0
            items = list(b)
            result = []
            a %= math.factorial(len(items))
            for x in range(len(items) - 1, -1, -1):
                fact = math.factorial(x)
                index = a // fact
                a -= index * fact
                result.append(items[index])
                del items[index]
            return result

        raise BadTypeCombinationError(".P", a, b)

class Eval_All_Input(Operator):
    arity = 0

    def operate(self):
        return [literal_eval(val) for val in all_input()]


class Round(Operator):
    arity = 2

    def operate(self, a, b):
        if is_num(a) and b == 0:
            return int(round(a))
        if is_num(a) and isinstance(b, int):
            return round(a, b)
        if is_num(a) and is_num(b):
            round_len = 0
            while round(b, round_len) != b and round_len < 15:
                round_len += 1
            return round(a, round_len)
        raise BadTypeCombinationError(".R", a, b)

class Shuffle(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, list):
            random.shuffle(a)
            return a
        if isinstance(a, str):
            tmp_list = list(a)
            random.shuffle(tmp_list)
            return ''.join(tmp_list)
        if is_col(a):
            tmp_list = list(a)
            random.shuffle(tmp_list)
            return tmp_list
        if is_num(a):
            tmp_list = urange(a)
            random.shuffle(tmp_list)
            return tmp_list

        raise BadTypeCombinationError('.S', a)

class Justified_Transpose(Operator):
    arity = 1

    def operate(self, a):
        if is_col(a):
            if not a:
                return a
            lol = [urange(elem) if is_num(elem) else elem for elem in a]
            cols = max(len(sublist) for sublist in lol)
            trans = [[] for _ in range(cols)]
            for sublist in lol:
                for index, elem in enumerate(sublist):
                    trans[index].append(elem)
            if all(isinstance(sublist, str) for sublist in lol):
                return list(map(''.join, trans))
            else:
                return list(map(list, trans))
        raise BadTypeCombinationError(".T", a)

class Reduce2(Lambda_Container):
    arity = 2
    params = ["b", "Z"]

    def operate(self, a, b):
        b = b.eval()
        if is_seq(b) or isinstance(b, int):
            if is_num(b):
                whole_seq = urange(b)
            else:
                whole_seq = b
            if len(whole_seq) == 0:
                raise BadTypeCombinationError(".U", a, b)

            acc = whole_seq[0]
            seq = whole_seq[1:]

            while len(seq) > 0:
                h = seq[0]
                acc = a(acc, h)
                seq = seq[1:]
            return acc
        raise BadTypeCombinationError(".U", a, b)

class Infinite_For(Control_Flow, Statement):
    block_from = 1

    def operate(self, a, *args):
        for value in infinite_iterator(a.eval()):
            Variable.env["b"] = value
            self.eval_block()

class Apply_While(Control_Flow):
    arity = 3

    params = [["H"], ["Z"]]

    def append_child(self, child):
        if len(self.children) < 2:
            return super().append_child(Lambda(self.params[len(self.children)])).append_child(child)
        return super().append_child(child)

    def operate(self, a, b, c):
        condition = a
        function = b
        value = c.eval()
        while condition(value):
            value = function(value)
        return value

class Zlib_Compress(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, str):
            a = a.encode('iso-8859-1')
            try:
                a = zlib.decompress(a)
            except:
                a = zlib.compress(a, 9)
            return a.decode('iso-8859-1')

        raise BadTypeCombinationError(".Z", a)

class Pad_Str(Operator):
    arity = 3

    def operate(self, a, b, c):
        if isinstance(a, str) and isinstance(b, str) and isinstance(c, int):
            pad_len = (c - len(a)) % c
            return a + (b * pad_len)[:pad_len]
        if isinstance(b, str) and isinstance(c, str) and isinstance(a, int):
            pad_len = (a - len(b)) % a
            pad_string = (c * pad_len)[:pad_len]
            return pad_string[:pad_len // 2] + b + pad_string[pad_len // 2:]
        if isinstance(c, str) and isinstance(a, str) and isinstance(b, int):
            pad_len = (b - len(c)) % b
            return (a * pad_len)[:pad_len] + c

        if is_seq(a) and isinstance(c, int):
            pad_len = (c - len(a)) % c
            return list(a) + [b] * pad_len
        if is_seq(b) and isinstance(a, int):
            pad_len = (a - len(b)) % a
            return [c] * (pad_len // 2) + list(b) + [c] * ((pad_len + 1) // 2)
        if is_seq(c) and isinstance(b, int):
            pad_len = (b - len(c)) % b
            return [a] * pad_len + list(c)

        raise BadTypeCombinationError(".[", a, b, c)

class Mod_Exp(Operator):
    arity = 3

    def operate(self, a, b, c):
        if isinstance(a, int) and isinstance(b, int) and isinstance(c, int):
            return pow(a, b, c)

        raise BadTypeCombinationError(".^", a, b, c)

class Prefixes(Operator):
    arity = 1

    def operate(self, a):
        if is_seq(a):
            return [a[:end] for end in range(1, len(a) + 1)]
        if is_num(a):
            if a < 0:
                return -1
            if a > 0:
                return 1
            else:
                return 0
        raise BadTypeCombinationError("._", a)

class Abs(Operator):
    arity = 1

    def operate(self, a):
        if is_num(a):
            return abs(a)
        if isinstance(a, tuple) or isinstance(a, list):
            if not a:
                return 0
            if is_num(a[0]):
                return sum(num ** 2 for num in a) ** .5
            if len(a) == 2:
                return sum((num1 - num2) ** 2 for num1, num2 in zip(*a)) ** .5

        raise BadTypeCombinationError(".a", a)

class Binary_Map(Lambda_Container):
    arity = 3
    params = ["N", "Y"]

    def operate(self, a, b, c=None):
        b = b.eval()
        c = c if c==None else c.eval()

        if c is None:
            b, c = zip(*b)
        if is_num(b):
            b = urange(b)
        if is_num(c):
            c = urange(c)
        if is_col(b) and is_col(c):
            return list(map(a, b, c))
        raise BadTypeCombinationError(".b", a, b, c)


class Combs(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            # compute n C r
            n, r = a, min(b, b - a)
            if r == 0:
                return 1
            if r < 0:
                r = b

            num = functools.reduce(operator.mul, range(n, n - r, -1), 1)
            den = math.factorial(r)

            return num // den

        if is_col(a) and isinstance(b, int):
            return itertools_norm(itertools.combinations, a, b)

        raise BadTypeCombinationError(".c", a, b)

class Datetime(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int):
            if a == 0:
                return time.time()
            if a == 1:
                return time.process_time()
            if 2 <= a <= 9:
                today = datetime.datetime.now()
                attributes = [today.year,
                              today.month,
                              today.day,
                              today.hour,
                              today.minute,
                              today.second,
                              today.microsecond]
                if a == 2:
                    return attributes
                if a < 9:
                    return attributes[a - 3]
                if a == 9:
                    return today.weekday()
        if is_num(a):
            time.sleep(a)
            return
        if is_col(a):
            return dict(a)
        raise BadTypeCombinationError(".d", a)

class Enumerated_Map(Lambda_Container):
    arity = 2
    params = ["b", "k"]

    def operate(self, a, b):
        b = b.eval()

        if is_col(b):
            return list(map(lambda arg: a(*arg), enumerate(b)))

        raise BadTypeCombinationError(".e", a, b)

class First_N(Lambda_Container):
    arity = 3
    params = ["Z"]

    def operate(self, a, b, c=None):
        b = b.eval()
        c = 1 if c == None else c.eval()

        if not isinstance(b, int):
            raise BadTypeCombinationError(".f", a, b, c)
        if is_num(c) or isinstance(c, str):
            return list(itertools.islice(filter(a, infinite_iterator(c)), b))
        elif is_col(c):
            return list(itertools.islice(filter(a, c), b))
        raise BadTypeCombinationError(".f", a, b, c)

class Group_By(Lambda_Container):
    arity = 2
    params = ["k"]

    def operate(self, a, b):
        b = b.eval()
        if is_num(b):
            seq = urange(b)
        elif is_col(b):
            seq = b
        else:
            raise BadTypeCombinationError(".g", a, b)
        key_sort = sorted(seq, key=a)
        grouped = itertools.groupby(key_sort, key=a)
        if isinstance(b, str):
            return list(map(lambda group: ''.join(group[1]), grouped))
        else:
            return list(map(lambda group: list(group[1]), grouped))

class Interleave(Operator):
    arity = 2

    def operate(self, a, b):
        if is_seq(a) and is_seq(b):
            overlap = min(len(a), len(b))
            longer = max((a, b), key=len)
            inter_overlap = [item for sublist in zip(a, b) for item in sublist]
            if isinstance(a, str) and isinstance(b, str):
                return ''.join(inter_overlap) + longer[overlap:]
            else:
                return inter_overlap + list(longer[overlap:])
        if is_col(a) and not is_seq(a):
            return interleave(sorted(list(a)), b)
        if is_col(b) and not is_seq(b):
            return interleave(a, sorted(list(b)))
        raise BadTypeCombinationError(".i", a, b)

class Complex(Operator):
    arity = 2

    def operate(self, a, b=1):
        if not is_num(a) and is_num(b):
            raise BadTypeCombinationError(".j", a, b)
        return a + b * complex(0, 1)

class Log(Operator):
    arity = 2

    def operate(self, a, b = math.e):
        if not is_num(a) or not is_num(b):
            raise BadTypeCombinationError(".l", a, b)
        if a < 0:
            return cmath.log(a, b)

        return math.log(a, b)

class Minimal(Lambda_Container):
    arity = 2
    params = ["b"]

    def operate(self, a, b):
        b = b.eval()
        if is_num(b):
            seq = urange(b)
        elif is_col(b):
            seq = b
        else:
            raise BadTypeCombinationError(".m", a, b)
        minimum = min(map(a, seq))
        return list(filter(lambda elem: a(elem) == minimum, seq))

class Numbers(Operator):
    arity = 1

    def operate(self, a):
        if isinstance(a, int) and a < 7:
            return [math.pi,
                    math.e,
                    2**.5,
                    (1 + 5**0.5) / 2,
                    float("inf"),
                    -float("inf"),
                    float("nan")][a]

        if is_lst(a):
            # Algorithm from:
            # http://stackoverflow.com/a/2158532/3739851
            # cc by-sa 3.0
            # Altered to use is_lst
            def flatten(l):
                for el in l:
                    if is_lst(el):
                        for sub in flatten(el):
                            yield sub
                    else:
                        yield el

            return list(flatten(a))

        raise BadTypeCombinationError(".n", a)

class Permutations(Operator):
    arity = 1

    def operate(self, a):
        if is_num(a):
            a = urange(a)
        if not is_col(a):
            raise BadTypeCombinationError(".p", a)
        return itertools_norm(itertools.permutations, a, len(a))

class Quit(Operator):
    arity = 0

    def operate(self):
        sys.exit(0)

class Rotate(Operator):
    arity = 2

    def operate(self, a, b):
        if is_col(a) and is_seq(b):
            def trans_func(elem):
                if elem in b:
                    elem_index = b.index(elem)
                    return b[(elem_index + 1) % len(b)]
                else:
                    return elem
            trans_a = map(trans_func, a)
            if isinstance(a, str):
                return ''.join(trans_a)
            if isinstance(a, set):
                return set(trans_a)
            return list(trans_a)

        raise BadTypeCombinationError(".r", a, b)

class Strip(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, str) and isinstance(b, str):
            return a.strip(b)
        if is_seq(a):
            if is_col(b):
                strip_items = list(b)
            else:
                strip_items = [b]
            seq = copy.deepcopy(a)
            while seq and seq[0] in strip_items:
                seq.pop(0)
            while seq and seq[-1] in strip_items:
                seq.pop()
            return seq
        raise BadTypeCombinationError(".s", a, b)

class Trig(Operator):
    arity = 2

    def operate(self, a, b):
        if is_num(a) and isinstance(b, int):

            funcs = [math.sin, math.cos, math.tan,
                     math.asin, math.acos, math.atan,
                     math.degrees, math.radians,
                     math.sinh, math.cosh, math.tanh,
                     math.asinh, math.acosh, math.atanh]

            return funcs[b](a)

        if is_lst(a):
            width = max(len(row) for row in a)
            padded_matrix = [list(row) + (width - len(row)) * [b] for row in a]
            transpose = list(zip(*padded_matrix))
            if all(isinstance(row, str) for row in a) and isinstance(b, str):
                normalizer = ''.join
            else:
                normalizer = list
            norm_trans = [normalizer(padded_row) for padded_row in transpose]
            return norm_trans
        raise BadTypeCombinationError(".t", a, b)

class Cumulative_Reduce(Lambda_Container):
    arity = 3
    params = ["N", "Y"]

    def operate(self, a, b, c = None):
        b = b.eval()
        c = c if c == None else c.eval()

        if c is None:
            counter = 0
            results = [copy.deepcopy(b)]
            acc = a(b, counter)
            while acc not in results:
                counter += 1
                results.append(copy.deepcopy(acc))
                acc = a(acc, counter)
            return results
        if is_seq(b) or is_num(b):
            if is_num(b):
                seq = urange(b)
            else:
                seq = b
            acc = c
            results = [copy.deepcopy(acc)]
            while len(seq) > 0:
                h = seq[0]
                acc = a(acc, h)
                seq = seq[1:]
                results.append(copy.deepcopy(acc))
            return results

class Write_File(Operator):
    arity = 2

    def operate(self, a, b = "o"):
        if not isinstance(b, str):
            raise BadTypeCombinationError(".w", a, b)

        if b.startswith("http"):
            if isinstance(a, dict):
                a = "&".join("=".join(i) for i in a.items())
            return [lin[:-1] if lin[-1] == '\n' else lin for lin
                    in urllib.request.urlopen(b, a.encode("UTF-8"))]

        prefix = b.split('.')[0] if b else 'o'
        suffix = b.split('.')[1] if '.' in b else None

        if is_lst(a):
            from PIL import Image
            suffix = suffix if suffix else 'png'

            if not is_lst(a[0][0]):
                a = [[(i, i, i) for i in j] for j in a]
            else:
                a = [[tuple(i) for i in j] for j in a]

            header = "RGBA" if len(a[0][0]) > 3 else "RGB"
            img = Image.new(header, (len(a[0]), len(a)))

            img.putdata(Psum(a))
            img.save(prefix + "." + suffix)
        else:
            suffix = suffix if suffix else "txt"

            with open(prefix + '.' + suffix, 'a', encoding='iso-8859-1') as f:
                if is_seq(a) and not isinstance(a, str):
                    f.write("\n".join(map(str, a)) + "\n")
                else:
                    f.write(str(a) + "\n")


class Try_Catch(Control_Flow):
    arity = 2

    def operate(self, a, b):
        try:
            return a.eval()
        except:
            return b.eval()

class All_Input(Operator):
    arity = 0

    def operate(self):
        return all_input()

class Set(Operator):
    arity = 1

    def operate(self, a=None):
        if a is None:
            return set()
        if is_num(a):
            return set([a])
        if is_col(a):
            try:
                return set(a)
            except TypeError:
                return set(map(tuple, a))
        raise BadTypeCombinationError("{", a)

class Bitwise_Or(Operator):
    arity = 2

    def operate(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a | b
        if is_col(a) and is_col(b):
            union = set(a) | set(b)
            if is_lst(a):
                return list(union)
            if isinstance(a, str):
                return ''.join(union)
            return union

        raise BadTypeCombinationError(".|", a, b)

############################################################

#
# class Unary_Map(Meta_Param):
#      arity = 1
#
#      def operate(self, a):
#          return [self.op.eval_with_children(i) for i in a]

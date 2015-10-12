from copy import deepcopy
from functools import reduce
import numbers
import collections

UNBOUNDED = -1

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

class Meta_Op(Operator):
	def __init__(self, op):
		super().__init__()
		self.op = op


	def __str__(self):
		return self.__class__.__name__ + "(" + str(self.op) + "): {" + ", ".join(map(str, self.children)) + "}"

class Root(Statement):
	arity = UNBOUNDED
	block_from = 0

	def eval(self):
		self.eval_block()

	def __init__(self):
		super().__init__()
		self.parent = self

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

	def operate(self, a):
		return a

class Imp_Print(Operator):
	arity = 1

	def operate(self, a):
		return print(a)

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
	arity = UNBOUNDED
	block_from = 1

	def operate(self, a, *args):
		self.true = False

		if a.eval():
			self.true = True
			self.eval_block()

class For_Loop(Control_Flow, Statement):
	arity = UNBOUNDED
	block_from = 2

	def operate(self, a, b, *args):
		for value in b.eval():
			variables[a.name] = value
			self.eval_block()

class While_Loop(Control_Flow, Statement):
	arity = UNBOUNDED
	block_from = 1

	def operate(self, a, *args):
		while a.eval():
			self.eval_block()

class Map(Lambda_Container):
	arity = 2
	params = ["d"]

	def operate(self, a, b):
		return [a.eval(value) for value in b.eval()]

class Fold(Meta_Op):
	arity = 1

	def operate(self, a):
		return reduce(self.op.operate, a, a.pop(0)) if a else 0

class Unary_Map(Meta_Op):
	 arity = 1

	 def operate(self, a):
		 return [self.op.operate(i) for i in a]

class Left_Map(Meta_Op):
	arity = 2

	def operate(self, a, b):
		return [self.op.operate(a, i) for i in b]

class Right_Map(Meta_Op):
	arity = 2

	def operate(self, a, b):
		return [self.op.operate(i, a) for i in b]

class Add(Operator):
	arity = 2

	def operate(self, a, b):
	    if isinstance(a, set):
	        if is_col(b):
	            return a.union(b)
	        else:
	            return a.union({b})
	    if is_lst(a) and not is_lst(b):
	        return list(a)+[b]
	    if is_lst(b) and not is_lst(a):
	        return [a]+list(b)
	    if is_lst(a) and is_lst(b):
	        return list(a)+list(b)
	    if is_num(a) and is_num(b) or\
	            isinstance(a, str) and isinstance(b, str):
	        return a+b
	    if is_num(a) and isinstance(b, str):
	        return str(a) + b
	    if isinstance(a, str) and is_num(b):
	        return a + str(b)
	    raise BadTypeCombinationError("+", a, b)


class Mul(Operator):
	arity = 2

	def operate(self, a, b):
		return a * b

class Sub(Operator):
	arity = 2

	def operate(self, a, b):
		return a - b

class Div(Operator):
	arity = 2

	def operate(self, a, b):
		if is_seq(a):
			return a.count(b)

		return a // b

class Float_Div(Operator):
	arity = 2

	def operate(self, a, b):
		return a / b

class List(Operator):
	arity = UNBOUNDED

	def operate(self, *args):
		return list(args)

class Couple(Operator):
	arity = 2

	def operate(self, a, b):
		return [a, b]

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

def to_base_ten(arb, base):
    # Special cases
    if abs(base) == 1:
        return len(arb)
    acc = 0
    for digit in arb:
        acc *= base
        acc += digit
    return acc

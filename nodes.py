from copy import deepcopy

UNBOUNDED = -1

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

class Root(Statement):
	arity = UNBOUNDED
	block_from = 0
	
	def eval(self):
		self.eval_block()
	
	def __init__(self):
		super().__init__()
		self.parent = self

class Number(Node):
	arity = 0
	
	def __init__(self, value):
		super().__init__()
		self.value = value
	
	def eval(self):
		return self.value

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
		"d": " "
}

class Variable(Node):
	arity = 0
	
	def __init__(self, name):
		super().__init__()
		self.name = name
	
	def eval(self):
		return variables[self.name]

class Assign(Control_Flow):
	arity = 2
	
	def operate(self, a, b):
		value = b.eval()
		
		variables[a.name] = value
		return value

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

class Add(Operator):
	arity = 2
	
	def operate(self, a, b):
		return a + b

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
		return a // b

class List(Operator):
	arity = UNBOUNDED
	
	def operate(self, *args):
		return args

class Couple(Operator):
	arity = 2
	
	def operate(self, a, b):
		return [a, b]

class Ternary(Control_Flow):
	arity = 3
	
	def operate(self, a, b, c):
		if a.eval():
			return b.eval()
		return c.eval()

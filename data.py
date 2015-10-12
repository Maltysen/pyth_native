from nodes import *

operators = {
	"+": Add,
	"*": Mul,
	"-": Sub,
	"/": Div,
	"c": Float_Div,
	"[": List,
	",": Couple,
	"]": One_List,
	"?": Ternary,
	" ": Suppress_Imp_Print,
	"I": If_Statement,
	"F": For_Loop,
	"W": While_Loop,
	"=": Assign,
	"~": Post_Assign,
	"m": Map,
	"s": Sum,
	"r": Range,
	"U": Urange,
	"q": Equals,
	"h": Head,
	"`": Repr,
	"C": Char
}

meta_ops = {
	"F": Fold,
	"M": Unary_Map,
	"L": Left_Map,
	"R": Right_Map
}

digits = ".0123456789"

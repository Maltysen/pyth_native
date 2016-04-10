from nodes import *

#TODO: \n $ '

operators = {
	" ": Suppress_Imp_Print,
	"!": Negate,
	"#": ErrorLoop,
	"%": Mod,
	"&": And,
	"(": Tuple,
	"*": Mul,
	"+": Plus,
	",": Couple,
	"-": Minus,
	"/": Div,

	"_": Neg,
	"c": Float_Div,
	"[": List,
	"]": One_List,
	"?": Ternary,
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
	"C": Char,
	"P": Primes_Pop
}

meta_ops = {
	"M": Map
}

digits = ".0123456789"

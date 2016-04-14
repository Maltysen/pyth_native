from nodes import *

#TODO: $
#TODO: '
#TODO: B
#TODO: D
#TODO: J
#TODO: K
#TODO: L
#TODO: M
#TODO: Q
#TODO: R
#TODO: z


operators = {
	"\n": Explicit_Print,
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
	":": Slice,
	"<": Less,
	"=": Assign,
	">": Greater,
	"?": Ternary,
	"@": Lookup,
	"A": Double_Assign,
	"C": Char,
	"E": Eval_Input,
	"F": For_Loop,
	"I": If_Statement,
	"O": Random,
	"P": Primes_Pop,
	"S": Sort,
	"U": Urange,
	"V": Implicit_For,
	"W": While_Loop,
	"X": Assign_At,
	"[": List,
	"]": One_List,
	"^": Exponentiate,
	"_": Opposite,
	"`": Repr,
	"a": Append,
	"c": Chop,
	"e": End,
	"f": Pfilter,
	"g": Greater_Equal,
	"h": Head,
	"i": Base10,
	"j": Join,
	"l": Length,
	"m": Map,
	"n": Not_Equal,
	"o": Order_By,
	"p": Debug_Print,
	"q": Equals,
	"r": Range,
	"s": Sum,
	"t": Tail,
	"u": Reduce,
	"v": Eval,
	"w": Raw_Input,
	"x": Index,
	"y": Powerset,
	"{": Uniquify,
	"|": Or,
	"}": In,
	"~": Post_Assign,
	".!": Factorial,
	".H": Hex
}

meta_ops = {
	"M": Map
}

digits = ".0123456789"

Variable.env = {
    "G": string.ascii_lowercase,
    "H": {},
    "N": '"',
    "Q": None,
    "T": 10,
    "Y": [],
    "Z": 0,
    "b": "\n",
    "d": " ",
    "k": "",
    "z": None
}

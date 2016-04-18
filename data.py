from nodes import *

#TODO: $
#TODO: '
#TODO: B
#TODO: D
#TODO: L
#TODO: M
#TODO: R
#TODO: ."
#TODO: .*
#TODO: .?
#TODO: .N
#TODO: .Q
#TODO: .w
#TODO: .z
#TODO: .v

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
	"f": Filter,
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
	".&": Bitwise_And,
	".(": Pop_Loc,
	".)": Pop,
	"._": Remove,
	"./": Partition,
	".:": Substrings,
	".<": Left_Shift,
	".>": Right_Shift,
	".A": All,
	".B": Bin_Str,
	".C": Combs_Repl,
	".D": Divmod,
	".E": Any,
	".F": Format_Str,
	".H": Hex_Str,
	".I": Invert,
	".M": Maximize,
	".O": Oct_Str,
	".P": Permutations2,
	".R": Round,
	".S": Shuffle,
	".T": Justified_Transpose,
	".U": Reduce2,
	".V": Infinite_For,
	".W": Apply_While,
	".Z": Zlib_Compress,
	".[": Pad_Str,
	".^": Mod_Exp,
	"._": Prefixes,
	".a": Abs,
	".b": Binary_Map,
	".c": Combs,
	".d": Datetime,
	".e": Enumerated_Map,
	".f": First_N,
	".g": Group_By,
	".i": Interleave,
	".j": Complex,
	".l": Log,
	".m": Minimal,
	".n": Numbers,
	".p": Permutations,
	".q": Quit,
	".r": Rotate,
	".s": Strip,
	".t": Trig,
	".u": Cumulative_Reduce,
	".x": Try_Catch,
	".{": Set,
	".|": Bitwise_Or
}

meta_ops = {
	"M": Map
}

digits = ".0123456789"

Variable.env = {
    "G": string.ascii_lowercase,
    "H": {},
	"K": None,
	"J": None,
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

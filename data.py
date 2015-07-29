from nodes import *

operators = {
	"+": (Add,),
	"*": (Mul,),
	"-": (Sub,),
	"/": (Div,),
	"[": (List,),
	",": (Couple,),
	"?": (Ternary,),
	" ": (Suppress_Imp_Print,),
	"I": (If_Statement,),
	"F": (For_Loop,),
	"=": (Assign,)
}

digits = ".0123456789"

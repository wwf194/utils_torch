import utils_torch
Operators = utils_torch.json.PyObj()

def Add(*Args):
    Sum = Args[0]
    for Index in range(1, len(Args)):
        Sum += Args[Index]
    return Sum
Operators.Add = Add

def Split(Arg):
    if isinstance(Arg, list):
        return Arg
    else:
        raise Exception
Operators.Split = Split

def CalculateGradient(self, loss):
    loss.backward()
Operators.CalculateGradient = CalculateGradient


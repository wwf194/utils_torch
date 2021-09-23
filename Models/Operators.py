import utils_torch
Operators = utils_torch.json.PyObj()

def Add(*Args):
    Sum = Args[0]
    for Index in range(1, len(Args)):
        Sum += Args[Index]
    return Sum

Operators.Add = Add
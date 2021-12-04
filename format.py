
import torch
import utils_torch
def ToNpArrayIfIsTensor(data):
    if isinstance(data, torch.Tensor):
        return utils_torch.ToNpArray(data), False
    else:
        return data, True

KB = 1024
MB = 1048576
GB = 1073741824
TB = 1099511627776

def ByteNum2Str(ByteNum):
    if ByteNum < KB:
        Str = "%d B"%ByteNum
    elif ByteNum < MB:
        Str = "%.3f KB"%(1.0 * ByteNum / KB)
    elif ByteNum < GB:
        Str = "%.3f MB"%(1.0 * ByteNum / MB)
    elif ByteNum < TB:
        Str = "%.3f GB"%(1.0 * ByteNum / GB)
    else:
        Str = "%.3f TB"%(1.0 * ByteNum / TB)
    return Str
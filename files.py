import os
import utils_torch
def RemoveAllFiles(path, verbose=True):
    if not os.path.exists(path):
        raise Exception()
    if not os.path.isdir(path):
        raise Exception()
    for file in ListAllFiles(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)
        utils_torch.add_log("utils_pytorch: removed file: %s"%file_path)

def ListAllFiles(path):
    if not os.path.exists(path):
        raise Exception()
    if not os.path.isdir(path):
        raise Exception()
    items = os.listdir(path)
    files = []
    for item in items:
        if os.path.isfile(os.path.join(path, item)):
            files.append(item)
    return files
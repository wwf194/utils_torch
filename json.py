
class PyJSON(object):
    def __init__(self, d=None):
        if d is not None:
            if type(d) is str:
                d = json.loads(d)
            self.from_dict(d)
    def from_dict(self, d):
        self.__dict__ = {}
        for key, value in d.items():
            if type(value) is dict:
                value = PyJSON(value)
            self.__dict__[key] = value
    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            if type(value) is PyJSON:
                value = value.to_dict()
            d[key] = value
        return d
    def __repr__(self):
        return str(self.to_dict())
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__[key]

def JsonObj2PythonObj(json_obj):
    obj = PyJSON()
    obj.from_dict(json_obj)
    return obj
    #return jsonpickle.decode(json_obj_to_json_str(json_obj))
    #return json.loads(json.dumps(dict), object_hook=lambda d: SimpleNamespace(**d))
json_obj_to_object = JsonObj2PythonObj

def JsonObj2JsonStr(json_obj):
    return json.dumps(json_obj)

json_obj_to_json_str = JsonObj2JsonStr

def PythonObj2JsonObj(obj):
    return json.loads(object_to_json_str(obj))

def PythonObj2JsonFile(obj, path):
    # json_obj = PythonObj2JsonObj(obj)
    # json_str = JsonObj2JsonStr(json_obj)
    json_str = PythonObj2JsonStr(obj)
    JsonStr2JsonFile(json_str, path)

def JsonStr2JsonFile(json_str, path):    
    if path.endswith(".jsonc") or path.endswith(".json"):
        pass
    else:
        path += ".jsnonc"
    with open(path, "w") as f:
        f.write(json_str)

object_to_json_obj = PythonObj2JsonObj

def PythonObj2JsonStr(obj):
    # return json.dumps(obj.__dict__, cls=change_type,indent=4)
    # why default=lambda o: o.__dict__?
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=False, indent=4)
object_to_json_str = PythonObj2JsonStr

def JsonStr2JsonObj(json_str):
    return json.loads(json_str)
json_str_to_json_obj = JsonStr2JsonObj

def JsonStr2PythonObj(json_str):
    json_obj = json_str_to_json_obj(json_str)
    return json_obj_to_object(json_obj)
    # return json.loads(json_str, object_hook=lambda d: SimpleNamespace(**d))
json_str_to_object = JsonStr2PythonObj

def JsonFile2JsonObj(file_path):
    with open(file_path, "r") as f:
        json_dict = json5.load(f)
    return json_dict
load_json_file = JsonFile2JsonObj

def JsonStr2JsonFile(json_str, path):
    with open(path, "w") as f:
        f.write(json_str)

new_json_file = JsonStr2JsonFile

def parse_param_json_dicts(json_dicts, overwrite=True):
    param = JsonObj2PythonObj(json_dicts)
    for name, obj in list_attrs(param):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    json_dicts_parsed = parse_obj(param)
    for value in json_dicts_parsed.values():
        value.pop("__DollarPath__")
    for name, obj in list_attrs(param):
        delattr(obj, "__DollarPath__")
    return json_dicts_parsed

def parse_json_obj(obj): # obj can either be dict or list.
    json_obj = obj
    obj = JsonObj2PythonObj(json_obj)
    parse_obj(obj)
    return PythonObj2JsonObj(obj)

JsonObj2ParsedJsonObj = parse_json_obj

def JsonObj2ParsedPythonObj(json_obj):
    return JsonObj2PythonObj(JsonObj2ParsedJsonObj(json_obj))

def parse_obj(obj):
    _parse_obj(obj, root=obj, attrs=[], parent=None)
    return object_to_json_obj(obj)

def _parse_obj(obj, root, attrs, parent, base_path="root."):
    if hasattr(obj, "__DollarPath__"):
        base_path = getattr(obj, "__DollarPath__")
    if isinstance(obj, list):
        for index, item in enumerate(obj):
            _parse_obj(item, root, attrs + [index], obj, base_path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _parse_obj(value, root, attrs + [key], obj, base_path)
    elif isinstance(obj, str):
        #if obj!="" and obj[0]=="$":
        if "$" in obj or "^" in obj:
            sentence = obj.replace("$", base_path).replace("^", "root.")
            #print(sentence)
            parent[attrs[-1]] = eval(sentence)
    elif isinstance(obj, object) and hasattr(obj, "__dict__"):
        for attr, value in obj.__dict__.items():
            _parse_obj(value, root, attrs + [attr], obj, base_path)
    else:
        #raise Exception("_parse_json_obj: Invalid type: %s"%type(obj))
        pass
    return obj

'''
def build_mlp(dict_):
    act_func = get_act_func_module(dict_["act_func"])
    layers = []
    layer_dicts = []
    unit_nums = dict_["unit_nums"] #input_num, hidden_layer1_unit_num, hidden_layer2_unit_numm ... output_num
    layer_num = len(unit_nums) - 1
    for layer_index in range(layer_num):
        current_layer = nn.Linear(unit_nums[layer_index], unit_nums[layer_index+1], bias=dict["bias"])
        layers.append(current_layer)
        layer_dicts.append(current_layer.state_dict())
        if not (dict_["act_func_on_last_layer"] and layer_index==layer_num-1):
            layers.append(act_func)
    dict["layer_dicts"] = layer_dicts
    return torch.nn.Sequential(*layers)
'''
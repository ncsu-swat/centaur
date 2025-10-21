
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def bitwise_and_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    other1 = torch.tensor([4, 5, 6], dtype=torch.int32).numpy()
    out1 = np.empty((3,), dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int32).numpy()
    other2 = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int32).numpy()
    out2 = np.empty((2, 3), dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([15, 7, 31], dtype=torch.int32).numpy()
    other3 = torch.tensor([8, 10, 15], dtype=torch.int32).numpy()
    out3 = np.empty((3,), dtype=np.int32)
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs["torch.bitwise_and"] = bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_and'.")


check_valid('torch.bitwise_and', generated_inputs['torch.bitwise_and'], lib="torch", suffix=0)

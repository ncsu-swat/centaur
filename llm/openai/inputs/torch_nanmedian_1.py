
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nanmedian_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0])
    list_of_inputs.append({"input": input1})
    input2 = np.array([1.0, np.nan, 3.0, 2.0])
    list_of_inputs.append({"input": input2})
    input3 = np.array([[1.0, 2.0], [3.0, np.nan]])
    list_of_inputs.append({"input": input3})
    input4 = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    list_of_inputs.append({"input": input4})
    input5 = np.array([np.nan, np.nan, np.nan])
    list_of_inputs.append({"input": input5})
    input6 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    list_of_inputs.append({"input": input6})
    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, np.nan], [7.0, 8.0]]])
    list_of_inputs.append({"input": input7})
    input8 = np.array([[np.nan, 2.0], [3.0, np.nan]])
    list_of_inputs.append({"input": input8})
    input9 = np.array([0.0, 0.0, 0.0])
    list_of_inputs.append({"input": input9})
    input10 = np.array([1.5, 2.5, 3.5, np.nan])
    list_of_inputs.append({"input": input10})
    return list_of_inputs

generated_inputs["torch.nanmedian_1"] = nanmedian_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nanmedian_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nanmedian_1'.")


check_valid('torch.nanmedian', generated_inputs['torch.nanmedian_1'], lib="torch", suffix=1)

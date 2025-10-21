
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def as_strided_copy_inputs():
    list_of_inputs = []
    
    source1 = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.int32)
    size1 = (2, 2, 2)
    stride1 = (4, 2, 1)
    storage_offset1 = 0
    out1 = np.empty((2, 2, 2)).astype(np.int32)

    input_dict1 = {
        "source": source1,
        "size": size1,
        "stride": stride1,
        "storage_offset": storage_offset1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    source2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64)
    size2 = (3, 2)
    stride2 = (2, 1)
    storage_offset2 = 0
    out2 = np.empty((3, 2)).astype(np.float64)

    input_dict2 = {
        "source": source2,
        "size": size2,
        "stride": stride2,
        "storage_offset": storage_offset2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    source3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).astype(np.int16)
    size3 = (2, 3, 2)
    stride3 = (6, 2, 1)
    storage_offset3 = 0
    out3 = np.empty((2, 3, 2)).astype(np.int16)

    input_dict3 = {
        "source": source3,
        "size": size3,
        "stride": stride3,
        "storage_offset": storage_offset3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    source4 = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    size4 = (2, 2)
    stride4 = (1, 1)
    storage_offset4 = 0
    out4 = np.empty((2, 2)).astype(np.float32)

    input_dict4 = {
        "source": source4,
        "size": size4,
        "stride": stride4,
        "storage_offset": storage_offset4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    source5 = np.array([1, 2, 3, 4, 5]).astype(np.int64)
    size5 = (5,)
    stride5 = (1,)
    storage_offset5 = 0
    out5 = np.empty((5,)).astype(np.int64)

    input_dict5 = {
        "source": source5,
        "size": size5,
        "stride": stride5,
        "storage_offset": storage_offset5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.as_strided_copy"] = as_strided_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.as_strided_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_strided_copy'.")


check_valid('torch.as_strided_copy', generated_inputs['torch.as_strided_copy'], lib="torch", suffix=0)

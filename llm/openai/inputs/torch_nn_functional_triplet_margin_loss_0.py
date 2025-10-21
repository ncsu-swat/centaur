
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def triplet_margin_loss_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(5, 128).astype(np.float32)
    input2 = np.random.rand(5, 128).astype(np.float32)
    input3 = np.random.rand(5, 128).astype(np.float32)
    margin1 = 0.5
    p1 = 2.0
    eps1 = 1e-6
    swap1 = True
    reduction1 = "mean"
    
    input_dict1 = {
        "anchor": input1,
        "positive": input2,
        "negative": input3,
        "margin": margin1,
        "p": p1,
        "eps": eps1,
        "swap": swap1,
        "reduction": reduction1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(10, 256).astype(np.float32)
    input3 = np.random.rand(10, 256).astype(np.float32)
    input4 = np.random.rand(10, 256).astype(np.float32)
    margin2 = 1.0
    p2 = 1.0
    eps2 = 1e-8
    swap2 = False
    reduction2 = "sum"
    
    input_dict2 = {
        "anchor": input2,
        "positive": input3,
        "negative": input4,
        "margin": margin2,
        "p": p2,
        "eps": eps2,
        "swap": swap2,
        "reduction": reduction2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input5 = np.random.rand(2, 64).astype(np.float32)
    input6 = np.random.rand(2, 64).astype(np.float32)
    input7 = np.random.rand(2, 64).astype(np.float32)
    margin3 = 0.2
    p3 = 3.0
    eps3 = 1e-7
    swap3 = True
    reduction3 = "none"
    
    input_dict3 = {
        "anchor": input5,
        "positive": input6,
        "negative": input7,
        "margin": margin3,
        "p": p3,
        "eps": eps3,
        "swap": swap3,
        "reduction": reduction3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input8 = np.random.rand(32, 512).astype(np.float32)
    input9 = np.random.rand(32, 512).astype(np.float32)
    input10 = np.random.rand(32, 512).astype(np.float32)
    margin4 = 0.75
    p4 = 2.5
    eps4 = 1e-9
    swap4 = False
    reduction4 = "mean"
    
    input_dict4 = {
        "anchor": input8,
        "positive": input9,
        "negative": input10,
        "margin": margin4,
        "p": p4,
        "eps": eps4,
        "swap": swap4,
        "reduction": reduction4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input11 = np.random.rand(64, 10).astype(np.float32)
    input12 = np.random.rand(64, 10).astype(np.float32)
    input13 = np.random.rand(64, 10).astype(np.float32)
    margin5 = 1.5
    p5 = 1.0
    eps5 = 1e-5
    swap5 = True
    reduction5 = "sum"

    input_dict5 = {
        "anchor": input11,
        "positive": input12,
        "negative": input13,
        "margin": margin5,
        "p": p5,
        "eps": eps5,
        "swap": swap5,
        "reduction": reduction5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input14 = np.random.rand(1, 3).astype(np.float32)
    input15 = np.random.rand(1, 3).astype(np.float32)
    input16 = np.random.rand(1, 3).astype(np.float32)
    margin6 = 0.1
    p6 = 2.0
    eps6 = 1e-10
    swap6 = False
    reduction6 = "none"

    input_dict6 = {
        "anchor": input14,
        "positive": input15,
        "negative": input16,
        "margin": margin6,
        "p": p6,
        "eps": eps6,
        "swap": swap6,
        "reduction": reduction6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input17 = np.random.rand(8, 64).astype(np.float32)
    input18 = np.random.rand(8, 64).astype(np.float32)
    input19 = np.random.rand(8, 64).astype(np.float32)
    margin7 = 0.9
    p7 = 3.0
    eps7 = 1e-4
    swap7 = True
    reduction7 = "mean"
    
    input_dict7 = {
        "anchor": input17,
        "positive": input18,
        "negative": input19,
        "margin": margin7,
        "p": p7,
        "eps": eps7,
        "swap": swap7,
        "reduction": reduction7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input20 = np.random.rand(16, 128).astype(np.float32)
    input21 = np.random.rand(16, 128).astype(np.float32)
    input22 = np.random.rand(16, 128).astype(np.float32)
    margin8 = 0.3
    p8 = 1.5
    eps8 = 1e-7
    swap8 = False
    reduction8 = "sum"

    input_dict8 = {
        "anchor": input20,
        "positive": input21,
        "negative": input22,
        "margin": margin8,
        "p": p8,
        "eps": eps8,
        "swap": swap8,
        "reduction": reduction8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input23 = np.random.rand(4, 32).astype(np.float32)
    input24 = np.random.rand(4, 32).astype(np.float32)
    input25 = np.random.rand(4, 32).astype(np.float32)
    margin9 = 1.2
    p9 = 2.2
    eps9 = 1e-6
    swap9 = True
    reduction9 = "none"

    input_dict9 = {
        "anchor": input23,
        "positive": input24,
        "negative": input25,
        "margin": margin9,
        "p": p9,
        "eps": eps9,
        "swap": swap9,
        "reduction": reduction9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input26 = np.random.rand(3, 64).astype(np.float32)
    input27 = np.random.rand(3, 64).astype(np.float32)
    input28 = np.random.rand(3, 64).astype(np.float32)
    margin10 = 0.6
    p10 = 2.8
    eps10 = 1e-8
    swap10 = False
    reduction10 = "mean"

    input_dict10 = {
        "anchor": input26,
        "positive": input27,
        "negative": input28,
        "margin": margin10,
        "p": p10,
        "eps": eps10,
        "swap": swap10,
        "reduction": reduction10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.triplet_margin_loss"] = triplet_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.triplet_margin_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.triplet_margin_loss'.")


check_valid('torch.nn.functional.triplet_margin_loss', generated_inputs['torch.nn.functional.triplet_margin_loss'], lib="torch", suffix=0)

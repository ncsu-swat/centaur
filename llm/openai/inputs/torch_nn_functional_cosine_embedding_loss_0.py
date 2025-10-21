
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def cosine_embedding_loss_inputs():
    list_of_inputs = []

    input1 = np.random.rand(5, 128).astype(np.float32)
    input2 = np.random.rand(5, 128).astype(np.float32)
    target = np.array([1, 1, -1, -1, 1], dtype=np.float32)
    margin = 0.2
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(10, 64).astype(np.float32)
    input2 = np.random.rand(10, 64).astype(np.float32)
    target = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=np.float32)
    margin = 0.5
    size_average = False
    reduce = False
    reduction = 'sum'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(2, 32).astype(np.float32)
    input2 = np.random.rand(2, 32).astype(np.float32)
    target = np.array([1, -1], dtype=np.float32)
    margin = 0.1
    size_average = True
    reduce = True
    reduction = 'none'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(8, 256).astype(np.float32)
    input2 = np.random.rand(8, 256).astype(np.float32)
    target = np.random.choice([-1, 1], size=8, p=[0.5, 0.5]).astype(np.float32)
    margin = 0.3
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(1, 512).astype(np.float32)
    input2 = np.random.rand(1, 512).astype(np.float32)
    target = np.array([1], dtype=np.float32)
    margin = 0.4
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(3, 10).astype(np.float32)
    input2 = np.random.rand(3, 10).astype(np.float32)
    target = np.array([-1, 1, -1], dtype=np.float32)
    margin = 0.6
    size_average = False
    reduce = False
    reduction = 'sum'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(7, 32).astype(np.float32)
    input2 = np.random.rand(7, 32).astype(np.float32)
    target = np.random.choice([-1, 1], size=7, p=[0.2, 0.8]).astype(np.float32)
    margin = 0.7
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(4, 64).astype(np.float32)
    input2 = np.random.rand(4, 64).astype(np.float32)
    target = np.array([1, 1, -1, -1], dtype=np.float32)
    margin = 0.8
    size_average = True
    reduce = True
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(6, 16).astype(np.float32)
    input2 = np.random.rand(6, 16).astype(np.float32)
    target = np.array([-1, -1, 1, 1, -1, 1], dtype=np.float32)
    margin = 0.9
    size_average = False
    reduce = True
    reduction = 'mean'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(9, 128).astype(np.float32)
    input2 = np.random.rand(9, 128).astype(np.float32)
    target = np.random.choice([-1, 1], size=9, p=[0.3, 0.7]).astype(np.float32)
    margin = 1.0
    size_average = True
    reduce = False
    reduction = 'none'

    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.cosine_embedding_loss"] = cosine_embedding_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.cosine_embedding_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cosine_embedding_loss'.")


check_valid('torch.nn.functional.cosine_embedding_loss', generated_inputs['torch.nn.functional.cosine_embedding_loss'], lib="torch", suffix=0)

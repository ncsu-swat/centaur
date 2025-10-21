
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def poisson_nll_loss_inputs():
    list_of_inputs = []

    input_dict_1 = {
        "log_input": True,
        "full": False,
        "size_average": True,
        "eps": 1e-08,
        "reduce": True,
        "reduction": "mean",
        "input": torch.randn(5, 2).numpy(),
        "target": torch.randint(0, 10, (5, 2)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {
        "log_input": False,
        "full": True,
        "size_average": False,
        "eps": 1e-05,
        "reduce": False,
        "reduction": "sum",
        "input": torch.randn(3, 4, 5).numpy(),
        "target": torch.randint(1, 20, (3, 4, 5)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {
        "log_input": True,
        "full": False,
        "size_average": True,
        "eps": 1e-09,
        "reduce": True,
        "reduction": "none",
        "input": torch.randn(2, 2).numpy(),
        "target": torch.randint(0, 5, (2, 2)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {
        "log_input": False,
        "full": True,
        "size_average": False,
        "eps": 1e-07,
        "reduce": False,
        "reduction": "mean",
        "input": torch.randn(10,).numpy(),
        "target": torch.randint(0, 10, (10,)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_dict_5 = {
        "log_input": True,
        "full": True,
        "size_average": True,
        "eps": 1e-06,
        "reduce": True,
        "reduction": "sum",
        "input": torch.randn(4, 3, 2, 1).numpy(),
        "target": torch.randint(0, 10, (4, 3, 2, 1)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_dict_6 = {
        "log_input": False,
        "full": False,
        "size_average": True,
        "eps": 1e-08,
        "reduce": True,
        "reduction": "mean",
        "input": torch.randn(1, 1).numpy(),
        "target": torch.randint(0, 10, (1, 1)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_dict_7 = {
        "log_input": True,
        "full": True,
        "size_average": False,
        "eps": 1e-05,
        "reduce": False,
        "reduction": "none",
        "input": torch.randn(6, 6).numpy(),
        "target": torch.randint(0, 10, (6, 6)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_dict_8 = {
        "log_input": False,
        "full": False,
        "size_average": True,
        "eps": 1e-09,
        "reduce": True,
        "reduction": "mean",
        "input": torch.randn(2, 3, 4, 5).numpy(),
        "target": torch.randint(0, 10, (2, 3, 4, 5)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    input_dict_9 = {
        "log_input": True,
        "full": True,
        "size_average": False,
        "eps": 1e-07,
        "reduce": False,
        "reduction": "sum",
        "input": torch.randn(7,).numpy(),
        "target": torch.randint(0, 10, (7,)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_dict_10 = {
        "log_input": False,
        "full": True,
        "size_average": True,
        "eps": 1e-06,
        "reduce": True,
        "reduction": "mean",
        "input": torch.randn(2, 2, 2, 2).numpy(),
        "target": torch.randint(0, 10, (2, 2, 2, 2)).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.nn.PoissonNLLLoss"] = poisson_nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.PoissonNLLLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.PoissonNLLLoss'.")


check_valid('torch.nn.PoissonNLLLoss', generated_inputs['torch.nn.PoissonNLLLoss'], lib="torch", suffix=0)

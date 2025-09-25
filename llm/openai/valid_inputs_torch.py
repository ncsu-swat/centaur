generated_inputs = {}
import torch, copy, numpy as np

def acos_inputs():
    list_of_inputs = []
    input = np.array([0.0, 1.0, -1.0]).astype(np.float32)
    list_of_inputs.append({"input": input})
    return list_of_inputs

generated_inputs["torch.acos_"] = acos_inputs()

import torch, copy

def floor__inputs():
    list_of_inputs = []
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([1.5, 2.7, 3.2]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([-0.5, -1.5, -2.5]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([[[ -1.0, -2.0], [-3.0, -4.0]], [[-5.0, -6.0], [-7.0, -8.0]]]).numpy()
    list_of_inputs.append({"input": input})

    input = torch.tensor([1.0]).numpy()
    list_of_inputs.append({"input": input})
    
    return list_of_inputs

generated_inputs["torch.floor_"] = floor__inputs()

import torch, copy
import numpy as np

def get_rng_state_inputs():
    list_of_inputs = []

    input1 = torch.tensor(1).numpy()
    list_of_inputs.append({"torch.get_rng_state": input1})

    input2 = torch.tensor(0).numpy()
    list_of_inputs.append({"torch.get_rng_state": input2})

    input3 = torch.tensor(-1).numpy()
    list_of_inputs.append({"torch.get_rng_state": input3})

    input4 = torch.tensor(100).numpy()
    list_of_inputs.append({"torch.get_rng_state": input4})

    input5 = torch.tensor([1, 2, 3]).numpy()
    list_of_inputs.append({"torch.get_rng_state": input5})

    input6 = torch.tensor([0, 0, 0]).numpy()
    list_of_inputs.append({"torch.get_rng_state": input6})
    
    input7 = torch.tensor([-1, -2, -3]).numpy()
    list_of_inputs.append({"torch.get_rng_state": input7})
    
    input8 = torch.randn(2, 3).numpy()
    list_of_inputs.append({"torch.get_rng_state": input8})
    
    input9 = torch.zeros(3, 4, 5).numpy()
    list_of_inputs.append({"torch.get_rng_state": input9})

    input10 = torch.ones((5,)).numpy()
    list_of_inputs.append({"torch.get_rng_state": input10})

    input11 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    list_of_inputs.append({"torch.get_rng_state": input11})

    return list_of_inputs

generated_inputs["torch.get_rng_state"] = get_rng_state_inputs()


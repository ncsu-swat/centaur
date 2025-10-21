generated_inputs = {}
import torch, copy, numpy as np

def abs_inputs():
    list_of_inputs = []

    input1 = np.array([-1, -2, -3])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 2, 3])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-1.5, 2.5, -3.5])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, -2], [-3, 4]])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, -2], [-3, 4]], [[5, -6], [-7, 8]]])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1]])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([0, 0, 0])
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([-1, 2, -3, 4, -5])
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    input11 = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    input_dict11 = {"input": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs["torch.abs_"] = abs_inputs()

import torch, copy
import numpy as np

def acos_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.1, 0.2, 0.3])
    out1 = np.array([])
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-0.1, -0.2, -0.3])
    out2 = np.array([])
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, 0.0, -1.0])
    out3 = np.array([])
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0.1, 0.2], [0.3, 0.4]])
    out4 = np.array([])
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0.9, 0.8, 0.7, 0.6])
    out5 = np.array([])
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([-0.9, -0.8, -0.7, -0.6])
    out6 = np.array([])
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.0])
    out7 = np.array([])
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([0.5, -0.5, 0.0])
    out8 = np.array([])
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([np.cos(np.pi/3), np.cos(np.pi/4), np.cos(np.pi/6)])
    out9 = np.array([])
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 2, 2)
    out10 = np.array([])
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.acos"] = acos_inputs()

import torch, copy
import numpy as np

def acos_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.0, 0.5, 1.0])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, 0.0, 1.0])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.2, -0.7, 0.9])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0.0, 0.5], [1.0, -1.0]])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[0.0, 0.5, 1.0], [-1.0, 0.2, 0.8]]])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([-1.0])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([0.0, 0.0, 0.0])
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([0.99, 0.999, 0.9999])
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([-0.99, -0.999, -0.9999])
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.acos_"] = acos_inputs()

import torch, copy
import numpy as np

def acosh_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    out1 = np.array([])
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.1, 1.2], [1.3, 1.4]])
    out2 = np.array([[0.0, 0.0], [0.0, 0.0]])
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([1.0001, 1.0002, 1.0003])
    out3 = np.array([])
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([2.0, 3.0, 4.0, 5.0])
    out4 = np.array([])
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1.5], [2.5], [3.5]])
    out5 = np.array([[0.0], [0.0], [0.0]])
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, np.inf, 2.0])
    out6 = np.array([])
    input_dict6 = {"input": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.001, 1.002, 1.003, 1.004])
    out7 = np.array([])
    input_dict7 = {"input": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.2, 1.8], [2.1, 2.7]])
    out8 = np.array([[0.0, 0.0], [0.0, 0.0]])
    input_dict8 = {"input": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 1.0, 1.0, 1.0])
    out9 = np.array([])
    input_dict9 = {"input": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1.7, 2.3, 2.9, 3.5])
    out10 = np.array([])
    input_dict10 = {"input": input10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.acosh"] = acosh_inputs()


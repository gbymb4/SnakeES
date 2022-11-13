import torch
import utils.torch_util
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


from policy.wf_model import WFPolicy
from utils import torch_util

set_seed(1)
model = WFPolicy(8, 1, True, False)
model.norm_init()

param_flat = torch_util.get_flatten_params(model)
print(param_flat)

torch_util.set_flatten_params(param_flat['params'], param_flat['lengths'], model)
print(model.get_param_list())

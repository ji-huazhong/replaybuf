import random
from abc import ABC

import numpy as np
import torch
is_npu_available = True
try:
    import torch_npu
except ImportError:
    is_npu_available = False


class DeepSpeedStrategy(ABC):
    
    def __init__(
        self,
        seed: int = 42,
        zero_stage=2,
        bf16=True,
        args=None,     
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.bf16 = bf16
        self.seed = seed

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_npu_available:
            torch.npu.manual_seed_all(seed)


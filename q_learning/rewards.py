import numpy as np

from typing import List

def centralized_reward(sinr_mue: float, sinr_d2ds: List[float]):
    mue_contrib = np.log2(1 + sinr_mue)
    d2d_contrib = sum([np.log2(1 + s) for s in sinr_d2ds])
    reward = mue_contrib + d2d_contrib
    return reward, mue_contrib, d2d_contrib
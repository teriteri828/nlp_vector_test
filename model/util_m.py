from dataclasses import dataclass
import numpy as np


@dataclass
class CosSimilar:
    def execute(self, v1, v2):
        if np.all(v1==0) or np.all(v2==0):
            return 0.0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

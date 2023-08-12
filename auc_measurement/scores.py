"""Container for holding and serializing/deserializing model prediction scores."""

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
import numpy as np


@dataclass_json
@dataclass
class Scores:
    auc: float
    f1: float
    accuracy: float
    roc_fpr: List[np.ndarray]
    roc_tpr: List[np.ndarray]
    roc_thresholds: List[np.ndarray]

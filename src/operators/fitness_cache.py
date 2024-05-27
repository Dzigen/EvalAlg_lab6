from typing import List
import numpy as np

from numba.experimental import jitclass 
from numba import int64, float64    # import the types
spec = [
    ('best_score', float64),
    ('global_optimum', int64),
    ('best_solution', float64[:]),
    ('age_limit', int64)
]

@jitclass(spec)
class FitnessCache:
    def __init__(self) -> None:
        self.best_score: float = 0
        self.global_optimum: float = 10
        self.best_solution: List[float]
        self.age_limit: int = 0

    def update(self, fitnesses: List[float], solutions: List[List[float]]) -> None:
        for fit, sol in zip(fitnesses, solutions):
            #print(fit, sol)
            if self.best_score < fit:
                self.best_score = fit
                self.best_solution = np.copy(sol)
                self.age_limit = 0

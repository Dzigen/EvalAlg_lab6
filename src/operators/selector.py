from typing import List
import numpy as np
from numba.experimental import jitclass 
from numba import int64

spec = [
    ('pop_size', int64),
    ('cand_d', int64)
]

@jitclass(spec)
class MySelector:
    def __init__(self, population_size: int, dim: int) -> None:
        self.pop_size: int = population_size
        self.cand_d: int = dim

    def filter_population(self, population: List[List[int]], fitnesses: List[float]) -> List[List[int]]:
        sorted_idx = list(map(lambda v: v[0], sorted(list(zip(list(range(len(population))), fitnesses)), 
                                                     key=lambda v: v[1], reverse=True)))

        filtered_pop = np.zeros(shape=(self.pop_size, self.cand_d), dtype=np.float64)
        for i in range(len(filtered_pop)):
            filtered_pop[i] = np.copy(population[sorted_idx[i]])

        return filtered_pop
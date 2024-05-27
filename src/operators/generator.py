import typing
from typing import List, Dict
import random
import numpy as np

from numba.experimental import jitclass 
from numba import int64, float64     # import the types
from numba.typed import List as NumbaList
import math

spec = [
    ('cand_d', int64),
    ('pop_mltply', int64),
    ('gen_mode', int64)
]
@jitclass(spec)
class MyGenerator:

    def __init__(self, dimension: int, gen_mode: int, over_pop_multiply: int = 5) -> None:
        self.cand_d: int = dimension
        self.pop_mltply: int = over_pop_multiply
        self.gen_mode: int = gen_mode

    def gen_random_solution(self) -> List[float]:
        return np.random.uniform(-5, 5, size=self.cand_d)
    
    def dist(self, p1: List[float], p2: List[float]) -> float:
        return math.sqrt(sum([math.pow(v1-v2,2) for v1, v2 in zip(p1,p2)]))
    
    def get_population(self, pop_size: int) -> List[List[float]]:
        if self.gen_mode == 0: # 'uniform'
            population = np.ones(shape=(pop_size, self.cand_d), dtype=np.float64)
            for i in range(pop_size):
                population[i] = np.copy(self.gen_random_solution())
        
        elif self.gen_mode == 1: # 'max_dist'
            over_size = pop_size * self.pop_mltply 
            over_pop = [self.gen_random_solution() for _ in range(over_size)]

            distances = []
            for i, point_anchor in enumerate(over_pop):
                min_dist = min([self.dist(point_anchor, point_neighbor) for j, point_neighbor in enumerate(over_pop) if j!=i])
                distances.append(min_dist)

            ranged_points = sorted(list(zip(distances, list(range(over_size)))), 
                                key=lambda v: v[0], reverse=True)

            population = np.ones(shape=(pop_size, self.cand_d), dtype=np.float64)
            for i in range(pop_size):
                cand = over_pop[ranged_points[i][1]]
                population[i] = np.copy(cand)
        
        elif self.gen_mode == 2: # 'bordered'
            population = np.ones(shape=(pop_size, self.cand_d), dtype=np.float64)
            for i in range(pop_size):
                probs = np.random.uniform(0, 1, size=self.cand_d)
                population[i] = np.array([5.0 if p > 0.5 else -5.0 for p in probs], dtype=np.float64)
            
        else:
            raise KeyError
        
        return population

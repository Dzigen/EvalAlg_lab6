from typing_extensions import List
import random
from numba.experimental import jitclass 
from numba import int64, float64    # import the types
import numpy as np

spec = [
    ('cand_d', int64),
    ('pairs_num', int64),
    ('gene_opt_prob', float64)
]

@jitclass(spec)
class MyCrossover:
    
    def __init__(self, dimension: int, crossover_pairs: int, 
                 crossover_gene_operation_prob: float) -> None:
        self.cand_d: int = dimension
        self.pairs_num: int = crossover_pairs
        self.gene_opt_prob: float = crossover_gene_operation_prob

    def mate(self, population: List[List[float]]) -> List[float]:
        new_solutions = np.ones(shape=(self.pairs_num, self.cand_d), dtype=np.float64)

        for i in range(self.pairs_num):
            cand1_idx, cand2_idx = -1, -1
            while True:
                cand1_idx = random.randint(0, len(population)-1)
                cand2_idx = random.randint(0, len(population)-1)
                if cand1_idx != cand2_idx:
                    break

            solution = []
            for g1, g2 in zip(population[cand1_idx], population[cand2_idx]):
                if random.uniform(0, 1) > self.gene_opt_prob:
                    # Дискретный кроссовер
                    solution.append(g1 if random.uniform(0,1) > 0.5 else g2)
                else:
                    # Арифметический кроссовер
                    lmbda = random.uniform(0, 1)
                    solution.append((lmbda * g1) + ((1 - lmbda) * g2))

            new_solutions[i] = np.array(solution, dtype=np.float64)

        return new_solutions


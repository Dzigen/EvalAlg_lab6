import random
import numpy as np
from typing import List, Dict

from numba.experimental import jitclass 
from numba import int64, float64    # import the types

spec = [
    ('cand_d', int64),
    ('iterval_board', int64),
    ('gene_opt_prob', float64),
    ('gauss_sigma', float64),
    ('mutation_cand_prob', float64),
    ('mutation_gene_prob', float64),
    ('mode', int64)
]

@jitclass(spec)
class MyMutation:

    def __init__(self, dimension: int, mut_cand_prob: float, 
                 mut_gene_prob:float, mut_gene_opt_prob: float, 
                 gauss_sigma: float, mut_mode: int) -> None:
        self.cand_d: int = dimension
        self.iterval_board: int = 5
        self.mode: int = mut_mode
        
        self.gene_opt_prob: float = mut_gene_opt_prob
        self.gauss_sigma = gauss_sigma
        self.mutation_cand_prob: float = mut_cand_prob
        self.mutation_gene_prob: float = mut_gene_prob

    def apply(self, population: List[List[float]]) -> List[List[float]]:
        
        mutated_ids = []
        mutated_sols = []

        for c_idx in range(len(population)):
            # Случайно выбираем особь для мутации
            if random.uniform(0,1) < self.mutation_cand_prob:
                continue
            else:
                mutated_ids.append(c_idx)
                cur_sol = np.copy(population[c_idx])

            for g_idx in range(self.cand_d):
                # Cлучайно выбираем ген особи для мутации
                if random.uniform(0,1) < self.mutation_gene_prob:
                    continue

                # Случайно выбираем оператор мутации
                updated_gene = None
                if random.uniform(0,1) > self.gene_opt_prob:
                    # Гауссова свёртка
                    while True:
                        updated_gene = cur_sol[g_idx] + random.gauss(0, self.gauss_sigma)
                        if abs(updated_gene) <= self.iterval_board:
                            break
                else:
                    # Равномерная свёртка
                    while True:
                        updated_gene = cur_sol[g_idx] + random.uniform(-5, 5)
                        if abs(updated_gene) <= self.iterval_board:
                            break
                
                cur_sol[g_idx] = updated_gene
                
            mutated_sols.append(np.copy(cur_sol))

        #
        if self.mode == 0:
            mutated_pop = np.copy(population)
            for i, idx in enumerate(mutated_ids):
                mutated_pop[idx] = mutated_sols[i]

        elif self.mode == 1:
            mutated_pop = np.ones(shape=(len(mutated_sols), self.cand_d), dtype=np.float64)
            for i, sol in enumerate(mutated_sols):
                mutated_pop[i] = np.copy(sol)
            mutated_pop = np.concatenate((population, mutated_pop))

        else:
            raise ValueError
            
        return mutated_pop
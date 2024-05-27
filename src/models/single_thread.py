from src.operators.crossover import MyCrossover
from src.operators.generator import MyGenerator
from src.operators.mutation import MyMutation
from src.operators.selector import MySelector
from src.operators.fitness_cache import FitnessCache
from src.operators.fitness_function import FitnessFunction

from src.models.master_slave import MasterSlaveModel

import gc
from typing_extensions import List
from tqdm import tqdm
import numpy as np

##########################

class SingleThreadModel:
    def __init__(self, dim: int, pop_size: int, cross_pairs: int, cross_gene_opt_prob: float,
                 mut_gene_opt_prob: float, mut_gauss_sigma:float, 
                 mut_cand_prob: float, mut_gene_prob: float, 
                 gen_mode: int, mut_mode: int) -> None:
        self.dim = dim
        self.pop_size = pop_size
        self.cross_pairs = cross_pairs
        self.cross_gene_opt_prob = cross_gene_opt_prob
        self.mut_gene_opt_prob = mut_gene_opt_prob
        self.mut_gauss_sigma = mut_gauss_sigma
        self.mut_cand_prob = mut_cand_prob
        self.mut_gene_prob = mut_gene_prob
        self.gen_mode = gen_mode
        self.mut_mode = mut_mode

    def run(self, iters: int, enable_ms_model: bool = False, 
            complexity: int = 0, ff_age_limit: int = 250, verbose:bool=False, workers:int=3, 
            base_pop: List[List[float]] = None) -> List[float]:
        
        #print("Инициализация операторов эволюционного процесса")
        ff = FitnessFunction(self.dim, complexity)
        ff_cache = FitnessCache()
        generator = MyGenerator(self.dim, self.gen_mode)
        selection_oprt = MySelector(self.pop_size, self.dim)
        crossover_oprt = MyCrossover(self.dim, self.cross_pairs, self.cross_gene_opt_prob)
        mutate_oprt = MyMutation(self.dim, self.mut_cand_prob, self.mut_gene_prob, 
                                self.mut_gene_opt_prob, self.mut_gauss_sigma, self.mut_mode)
        
        #print("Инициализация популяции")
        if base_pop is None:
            base_pop = generator.get_population(self.pop_size)

        if enable_ms_model:
            ms_model = MasterSlaveModel(workers, ff)

        #print(f"Старт эволюционного процесса: {base_pop.shape} {base_pop.dtype}")
        best_fitnesses, flag = [], False
        process = tqdm(range(iters)) if verbose else range(iters)
        for iter_idx in process:

            #print(f"Проверка критерия останова: {base_pop.shape} {base_pop.dtype}")
            if ff_cache.best_score >= 9.9 or ff_cache.age_limit > ff_age_limit:
                if ff_cache.age_limit > ff_age_limit:
                    flag = True
                best_fitnesses.append(ff_cache.best_score)
                break
            
            #print(f"Мутация кандидатов: {base_pop.shape} {base_pop.dtype}")
            old_pop = mutate_oprt.apply(base_pop)

            #print(f"Кроссовер кандидатов: {old_pop.shape} {old_pop.dtype}")
            new_pop = crossover_oprt.mate(old_pop)

            union_pop = np.concatenate((old_pop,new_pop))
            if enable_ms_model:
                union_pop, union_fit = ms_model.update_fitness(union_pop)
            else:
                union_fit = np.array([ff.calculate_fitness(sol) for sol in union_pop], dtype=np.float64)

            ff_cache.update(union_fit, union_pop)

            #print(f"Селекция кандидатов: {union_pop.shape} {union_pop.dtype}")
            base_pop = selection_oprt.filter_population(union_pop, union_fit)
            if verbose:
                process.set_postfix({"best-ff": ff_cache.best_score})
            best_fitnesses.append(ff_cache.best_score)
            ff_cache.age_limit += 1

        best_iter = iter_idx+1-ff_age_limit if flag else iter_idx+1

        #
#        del ff
#        del ff_cache
#        del generator
#        del selection_oprt
#        del crossover_oprt
#        del mutate_oprt
#        if enable_ms_model:
#            del ms_model

#        del union_pop
#        del union_fit

#        gc.collect()

        return best_fitnesses, best_iter, base_pop
from typing import List
import multiprocessing
import numpy as np
from tqdm import tqdm
import time

from src.models.single_thread import SingleThreadModel
from src.operators.fitness_cache import FitnessCache
from src.operators.fitness_function import FitnessFunction

class IslandBasedModel:
    def __init__(self, dim: int, pop_size: int, cross_pairs: int, cross_gene_opt_prob: float,
                 mut_gene_opt_prob: float, mut_gauss_sigma:float, 
                 mut_cand_prob: float, mut_gene_prob: float, 
                 gen_mode: int, mut_mode: int, sleep_t: int = 3) -> None:
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

        self.sleep_t = sleep_t
        self.manager = multiprocessing.Manager()
        self.lock = multiprocessing.Lock()

    def create_island(self, i, complited_work, queued_work, flags, 
                      iters, n_islands, ff_age_limit, complexity) -> None:
        st_model = SingleThreadModel(self.dim, self.pop_size // n_islands, self.cross_pairs,
                                     self.cross_gene_opt_prob, self.mut_gene_opt_prob,
                                     self.mut_gauss_sigma, self.mut_cand_prob, self.mut_gene_prob,
                                     self.gen_mode, self.mut_mode)
        while not flags['end']:
            self.lock.acquire()
            base_pop = queued_work.pop() if len(queued_work) != 0 else -1
            self.lock.release()

            if type(base_pop) is int and  base_pop == -1:
                #print(f"Island {i} waiting")
                time.sleep(3)
            else:
                #print(f"Island {i} start")
                _, _, updated_pop = st_model.run(
                    iters, complexity=complexity, ff_age_limit=ff_age_limit, base_pop=base_pop)
                complited_work.append(updated_pop)
                #print(f"Island {i} ready")
                
    def observe_islands(self, complited_work, queued_work, final_result,
                        flags, n_islands, epochs, verbose):
        ff = FitnessFunction(self.dim)
        ff_cache = FitnessCache()

        base_populations = [None for _ in range(n_islands)]
        process = tqdm(range(epochs)) if verbose else range(epochs)
        best_fitnesses = []
        for _ in process:
            #print(f"Observer sheduling work")
            complited_work[:] = []
            self.lock.acquire()
            for pop in base_populations:
                queued_work.append(pop)
            self.lock.release()

            #print(f"Observer waiting")
            while len(complited_work) != n_islands:
                time.sleep(self.sleep_t)

            #print(f"Observer proceding result")
            population = np.concatenate(list(complited_work))
            np.random.shuffle(population)
            fit = np.array([ff.calculate_fitness(sol) for sol in population], dtype=np.float64)
            ff_cache.update(fit, population)

            base_populations = population.reshape(n_islands, self.pop_size // n_islands, self.dim)

            if verbose:
                process.set_postfix({"best-ff": ff_cache.best_score})
            best_fitnesses.append(ff_cache.best_score)

        flags['end'] = True
        final_result.append(best_fitnesses)
        final_result.append(ff_cache.best_solution)



    def run(self, epochs: int, iters: int, complexity: int = 0, 
            ff_age_limit: int = 250, verbose: bool = False, n_islands: int = 3):
        
        complited_work = self.manager.list([])
        queued_work = self.manager.list([])
        final_result = self.manager.list([])
        flags = self.manager.dict({'end' : False})

        #
        observer = multiprocessing.Process(target=self.observe_islands, args=(complited_work, queued_work, final_result, 
                                                                              flags, n_islands, epochs, verbose))
        observer.start()
        jobs = []
        for i in range(n_islands):
            jobs.append(multiprocessing.Process(target=self.create_island, args=(i, complited_work, queued_work, flags, 
                                                                                 iters, n_islands, ff_age_limit, complexity)))
            jobs[-1].start()

        #
        observer.join()
        for job in jobs:
            job.join()

        #
        best_solution = final_result.pop()
        best_fitnesses = final_result.pop()

        return best_fitnesses, best_solution
        
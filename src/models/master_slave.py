import multiprocessing
from typing import List, Tuple
from src.operators.fitness_function import FitnessFunction
import numpy as np
import time

class MasterSlaveModel:
    def __init__(self, num_workers: int, ff: FitnessFunction, sleep_t: int = 0.00001):
        self.manager = multiprocessing.Manager()
        self.lock = multiprocessing.Lock()
        self.n_workers = num_workers
        self.ff = ff

    def stop_workers(self):
        for job in self.jobs:
            job.join()

    def do_work(self, i, complited_sols, complited_fits, queued_work):
        while True:
            self.lock.acquire()
            solution = queued_work.pop() if len(queued_work) != 0 else None
            self.lock.release()

            if solution is None:
                break
            else:
                #print(f"Process {i} proceeding")
                fitness = self.ff.calculate_fitness(solution)
                self.lock.acquire()
                complited_sols.append(solution)
                complited_fits.append(fitness)
                self.lock.release()
                #print(f"Process {i} saving results")

    def update_fitness(self, solutions: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
        complited_sols = self.manager.list([]) 
        complited_fits = self.manager.list([])
        queued_work = self.manager.list(solutions)
        
        self.jobs = []
        for i in range(self.n_workers):
            self.jobs.append(multiprocessing.Process(target=self.do_work, args=(i, complited_sols, complited_fits,
                                                                                queued_work)))
            self.jobs[-1].start()

        self.stop_workers()

        sols = np.array(list(complited_sols), dtype=np.float64)
        fits = np.array(list(complited_fits), dtype=np.float64)

        return sols, fits
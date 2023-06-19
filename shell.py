#! bpython -i
import importlib
import os

print("loading data...")
import luto.simulation as s
import luto.solvers.solve_column_generation as scg
import luto.solvers.solver as solver
import cProfile

print(f"done. PID {os.getpid()}")

def reload():
    global s, scg, solver

    scg = importlib.reload(scg)
    solver = importlib.reload(solver)
    s = importlib.reload(s)

def run(cluster_size=100, resfactor=10):
    reload()
    s.run_example_sim(cluster_size=cluster_size, resfactor=resfactor)

import pandas as pd
from tqdm import tqdm

import dgp
import design
from plan import Plan
import estimator as est
import evaluator as evalr
import numpy as np

def make_plan(designs):
    plan = Plan()

    for name, dgn, estr in designs:
        plan.add_design(name, dgn, estr)

    plan.add_evaluator('ATEError', evalr.ATEError)
    plan.add_evaluator('ITEBias', evalr.ITEBias)
    plan.add_evaluator('ITEMSE', evalr.ITEMSE)
    plan.add_evaluator('CovariateMSE', evalr.CovariateMSE)
    plan.add_evaluator('ATECovers', evalr.ATECovers)
    plan.add_evaluator('CISize', evalr.CISize)
    return plan

NUM_ITERS = 100

plan = make_plan([
    ('QuickBlock-B', design.QuickBlock, est.Blocking),
    ('QuickBlock-RF', design.QuickBlock, est.BlockingRF),
    ('OptBlock-B', design.OptBlock, est.Blocking),
    ('OptBlock-RF', design.OptBlock, est.BlockingRF),
    ('Randomization-RF', design.Bernoulli, est.OLSandRFT),
    ('Fixed Margins Randomization-RF', design.Complete, est.OLSandRFT),
    ('Rerandomization-RF', design.ReRandomization, est.OLSandRFT),
    #('Rerandomization-KNN', design.ReRandomization, est.OLSandKNNT),
    ('Matched Pair-B', design.MatchedPair, est.MatchedPairBlocking),
    ('SoftBlock-L', design.SoftBlock, est.LaplacianNorm),
    ('SoftBlock-RF', design.SoftBlock, est.OLSandRFT),
    ('GreedyNeighbors-RF', design.GreedyNeighbors, est.OLSandRFT),
    ('GreedyNeighbors-L', design.GreedyNeighbors, est.LaplacianNorm),
    #('KallusHeuristic-KNN', design.Heuristic, est.DMandKNNT),
    ('KallusHeuristic-RF', design.Heuristic, est.DMandRFT),
    #('KallusPSOD-KNN', design.PSOD, est.DMandKNNT),
    ('KallusPSOD-RF', design.PSOD, est.DMandRFT),
])

dfs = []
dgp_factory_class_list = [dgp.LinearFactory, dgp.QuickBlockFactory, dgp.SinusoidalFactory, dgp.TwoCirclesFactory]
sample_sizes = [10, 20, 32, 50, 64, 72, 86, 100]
for sample_size in sample_sizes[::-1]:
    print(f"Sample Size: {sample_size}")
    dgp_factory_list = [factory(N=sample_size) for factory in dgp_factory_class_list]
    for dgp_factory in dgp_factory_list:
        dgp_name = type(dgp_factory.create_dgp()).__name__
        print(f"DGP name: {dgp_name}")
        for it in tqdm(range(NUM_ITERS)):
            result = plan.execute(dgp_factory, seed=it * 1001)
            result['iteration'] = it
            result['sample_size'] = sample_size
            result['dgp'] = dgp_name
            filename = f"results/{dgp_name}_n{sample_size}_i{it}.csv.gz"
            result.to_csv(filename, index=False)
            dfs.append(result)


plan = make_plan([
    ('QuickBlock-B', design.QuickBlock, est.Blocking),
    ('QuickBlock-RF', design.QuickBlock, est.BlockingRF),
    #('OptBlock-B', design.OptBlock, est.Blocking),
    ('OptBlock-RF', design.OptBlock, est.BlockingRF),
    ('SoftBlock-L', design.SoftBlock, est.LaplacianNorm),
    ('SoftBlock-RF', design.SoftBlock, est.OLSandRFT),
    ('GreedyNeighbors-RF', design.GreedyNeighbors, est.OLSandRFT),
    ('GreedyNeighbors-L', design.GreedyNeighbors, est.LaplacianNorm),
    ('Randomization-RF', design.Bernoulli, est.OLSandRFT),
    ('Fixed Margins Randomization-RF', design.Bernoulli, est.OLSandRFT),
    ('Rerandomization-RF', design.ReRandomization, est.OLSandRFT),
    #('Rerandomization-KNN', design.ReRandomization, est.OLSandKNNT),
    #('KallusPSOD-KNN', design.PSOD, est.DMandKNNT),
    ('KallusPSOD-RF', design.PSOD, est.DMandRFT),
])

sample_sizes = [128, 150, 200, 250, 500, 1000, 2000, 3000]
for sample_size in sample_sizes[::-1]:
    print(f"Sample Size: {sample_size}")
    dgp_factory_list = [factory(N=sample_size) for factory in dgp_factory_class_list]
    for dgp_factory in dgp_factory_list:
        dgp_name = type(dgp_factory.create_dgp()).__name__
        print(f"DGP name: {dgp_name}")
        for it in tqdm(range(NUM_ITERS)):
            result = plan.execute(dgp_factory, seed=it * 1001)
            result['iteration'] = it
            result['sample_size'] = sample_size
            result['dgp'] = dgp_name
            filename = f"results/{dgp_name}_n{sample_size}_i{it}.csv.gz"
            result.to_csv(filename, index=False)
            dfs.append(result)


plan = make_plan([
    ('SoftBlock-L', design.SoftBlock, est.LaplacianNorm),
    ('SoftBlock-RF', design.SoftBlock, est.OLSandRFT),
    ('GreedyNeighbors-RF', design.GreedyNeighbors, est.OLSandRFT),
    ('GreedyNeighbors-L', design.GreedyNeighbors, est.LaplacianNorm),
    ('QuickBlock-B', design.QuickBlock, est.Blocking),
    ('QuickBlock-RF', design.QuickBlock, est.BlockingRF),
    ('Randomization', design.Bernoulli, est.OLSandRFT),
    ('Fixed Margins Randomization-RF', design.Complete, est.OLSandRFT),
    ('Rerandomization-RF', design.ReRandomization, est.OLSandRFT),
    # ('Rerandomization-KNN', design.ReRandomization, est.OLSandKNNT),
])

NUM_ITERS = 25

sample_sizes = [4000, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 30000, 50000, 100000]
for sample_size in sample_sizes:
    print(f"Sample Size: {sample_size}")
    dgp_factory_list = [factory(N=sample_size) for factory in dgp_factory_class_list]
    for dgp_factory in dgp_factory_list:
        dgp_name = type(dgp_factory.create_dgp()).__name__
        print(f"DGP name: {dgp_name}")
        for it in tqdm(range(NUM_ITERS)):
            result = plan.execute(dgp_factory, seed=it * 1001)
            result['iteration'] = it
            result['sample_size'] = sample_size
            result['dgp'] = dgp_name
            filename = f"results/{dgp_name}_n{sample_size}_i{it}.csv.gz"
            result.to_csv(filename, index=False)
            dfs.append(result)


NUM_ITERS = 5

sample_sizes = [50000, 100000, 500000, 1000000]
for sample_size in sample_sizes:
    print(f"Sample Size: {sample_size}")
    dgp_factory_list = [factory(N=sample_size) for factory in dgp_factory_class_list]
    for dgp_factory in dgp_factory_list:
        dgp_name = type(dgp_factory.create_dgp()).__name__
        print(f"DGP name: {dgp_name}")
        for it in tqdm(range(NUM_ITERS)):
            result = plan.execute(dgp_factory, seed=it * 1001)
            result['iteration'] = it
            result['sample_size'] = sample_size
            result['dgp'] = dgp_name
            filename = f"results/{dgp_name}_n{sample_size}_i{it}.csv.gz"
            result.to_csv(filename, index=False)
            dfs.append(result)

results = pd.concat(dfs)

filename = f"results/all_results.csv.gz"

print(f"""
\n**********************************************************************
***\tSAVING TO `{filename}`\t\t   ***
**********************************************************************""")

results.to_csv(filename, index=False)

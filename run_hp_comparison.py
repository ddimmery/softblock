import pandas as pd
from tqdm import tqdm

import dgp
import design
from plan import Plan
import estimator as est
import evaluator as evalr
import numpy as np

NUM_ITERS = 50

def make_plan(designs):
    plan = Plan()

    for name, dgn, estr, kw in designs:
        plan.add_design(name, dgn, estr, kw)

    plan.add_evaluator('ATEError', evalr.ATEError)
    plan.add_evaluator('ITEBias', evalr.ITEBias)
    plan.add_evaluator('ITEMSE', evalr.ITEMSE)
    plan.add_evaluator('CovariateMSE', evalr.CovariateMSE)
    plan.add_evaluator('ATECovers', evalr.ATECovers)
    plan.add_evaluator('CISize', evalr.CISize)
    return plan

dfs = []
for s in np.logspace(-5, 1, num=10, base=2):
    print(f"sigma is: {s}")
    plan = make_plan([
        ('SoftBlock-L', design.SoftBlock, est.LaplacianNorm, {'s2': s ** 2}),
        ('SoftBlock-RF', design.SoftBlock, est.OLSandRFT, {'s2': s ** 2}),
        ('KallusHeuristic-RF', design.Heuristic, est.DMandRFT, {'kernel_kwargs': {'s': s}}),
        ('KallusPSOD-RF', design.PSOD, est.DMandRFT, {'kernel_kwargs': {'s': s}}),
    ])
    dgp_factory = dgp.TwoCirclesFactory(N=90)
    for it in tqdm(range(NUM_ITERS)):
        result = plan.execute(dgp_factory, seed=it * 1001)
        result['iteration'] = it
        result['s'] = s
        #filename = f"results/HP_n{sample_size}_i{it}.csv.gz"
        # result.to_csv(filename, index=False)
        dfs.append(result)

results = pd.concat(dfs)

filename = f"results/hyperparam_results.csv.gz"

print(f"""
\n**********************************************************************
***\tSAVING TO `{filename}`\t\t   ***
**********************************************************************""")

results.to_csv(filename, index=False)

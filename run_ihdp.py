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

NUM_ITERS = 10000


plan = make_plan([
    #('QuickBlock-B', design.QuickBlock, est.Blocking),
    ('QuickBlock-RF', design.QuickBlock, est.BlockingRF),
    #('OptBlock-B', design.OptBlock, est.Blocking),
    ('OptBlock-RF', design.OptBlock, est.BlockingRF),
    ('Randomization', design.Bernoulli, est.OLSandRFT),
    #('Fixed Margins Randomization-RF', design.Bernoulli, est.OLSandRFT),
    ('Rerandomization-RF', design.ReRandomization, est.OLSandRFT),
    #('Rerandomization-KNN', design.ReRandomization, est.OLSandKNNT),
    # ('Matched Pair', design.MatchedPair, est.MatchedPairBlocking),
    ('SoftBlock-RF', design.SoftBlock, est.LaplacianRF),
    #('SoftBlock-RF', design.SoftBlock, est.OLSandRFT),
    ('GreedyNeighbors-RF', design.GreedyNeighbors, est.LaplacianRF),
    #('GreedyNeighbors-L', design.GreedyNeighbors, est.LaplacianNorm),
    # ('KallusHeuristic-KNN', design.Heuristic, est.DMandKNNT),
    # ('KallusHeuristic-RF', design.Heuristic, est.DMandRFT),
    # ('KallusPSOD-KNN', design.PSOD, est.DMandKNNT),
    ('KallusPSOD-RF', design.PSOD, est.DMandRFT),
])

dfs = []
for sd_ite in np.linspace(0, 12.5, num=1):
    init_ite_sd = []
    dgp_factory = dgp.IHDPFactory(
        csv_path="/Users/drewd/Documents/GitHub/softblock/dgp/ihdp.csv",
        sd_ite=sd_ite
    )
    dgp_name = type(dgp_factory.create_dgp()).__name__
    print(f"DGP name: {dgp_name}, SD: {sd_ite * 0.75}")
    for it in tqdm(range(NUM_ITERS)):
        result = plan.execute(dgp_factory, seed=it * 1001)
        init_ite_sd.append(plan.dgp.init_sd_ite)
        result['iteration'] = it
        result['sample_size'] = sd_ite
        result['dgp'] = dgp_name
        filename = f"results/{dgp_name}_sd{sd_ite}_i{it}.csv.gz"
        result.to_csv(filename, index=False)
        dfs.append(result)
    print(f"Average initial ITE: {np.average(init_ite_sd)} ± {np.std(init_ite_sd)/np.sqrt(len(init_ite_sd))}")

pd.concat(dfs).to_csv('IHDP_full.csv.gz', index=False)

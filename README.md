# SoftBlock replication

This repository contains replication materials and implementations of the algorithms in the AISTATs 2021 paper ["Efficient Balanced Treatment Assignments for Experimentation" by David Arbour, Drew Dimmery and Anup Rao.](https://arxiv.org/abs/2010.11332)

# Table of Contents

- [Overview of Code](#overview-of-code)
  * [`demo-**/`](#-demo-----)
  * [`design/`](#-design--)
  * [`dgp/`](#-dgp--)
  * [`figures/`](#-figures--)
- [Direct replication](#direct-replication)
  * [Figure 1](#figure-1)
  * [Figures 2, 3 and 4](#figures-2--3-and-4)
  * [Figure 5](#figure-5)
  * [Figure 6](#figure-6)
- [R implementation](#r-implementation)
  * [Example randomization](#example-randomization)
- [Python implementation](#python-implementation)

# Overview of Code

## `demo-**/`

The folders contain the step-by-step demonstrations of how a variety of methods perform design on a given set of data.

## `design/`

This folder contains the implementations of all methods shown in the paper. These methods are:
- Bernoulli (simple) randomization
- Complete (fixed margins) randomization
- GreedyNeighbors (new in our paper): MAXCUT on the nearest-neighbor graph
- Kallus Heuristic (see [Kallus 2017](https://arxiv.org/abs/1312.0531))
- Kallus PSOD (see [Kallus 2017](https://arxiv.org/abs/1312.0531))
- Matched pairs 
- OptBlock (see [Greevy et al 2004](https://pubmed.ncbi.nlm.nih.gov/15054030/))
- QuickBlock (see [Higgins et al 2016](https://www.pnas.org/content/113/27/7369))
- Rerandomization (see, e.g. [Morgan et al 2012](https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-2/Rerandomization-to-improve-covariate-balance-in-experiments/10.1214/12-AOS1008.full))
- SoftBlock (new in our paper): MAXCUT on the maximal spanning tree

## `dgp/`

The simulated data generating processes considered in our simulation analyses:
- IHDP (a common causal benchmark)
- Linear (just a linear outcome model)
- QuickBlock (the product of two uniform random variables)
- Sinusoidal (the outcome is a sinusoidal function of covariates)
- Two Circles (the covariates are distributed uniformly in two concentric circles and the outcome is a linear function of the radius and angle of the point from the origin)

## `figures/`

The raw figures used in the paper.

# Direct replication

## Figure 1

For replication of each of the subfigures in Figure 1, run `create_example_data.py` to generate example data and then plot it using `plot_example.R`.

## Figures 2, 3 and 4

Run `run_test.py`. When it has completed, run `analysis.R`.

## Figure 5

Once `run_test.py` has completed, run `analysis.R`.

## Figure 6

Run `run_hp_comparison.py` followed by `hp_analysis.R`.

# R implementation

Additionally, we have provided an R implementation of SoftBlock and GreedyNeighbors using tidyverse semantics in `r_implementation.R`.

## Example randomization

An self-contained example randomization using precinct-level elections data is available in `r-demo.Rmd`.

The core of the implementation is to simply call:

```{r}
source("https://raw.githubusercontent.com/ddimmery/softblock/master/r_implementation.R")

df %>% assign_greedy_neighbors(c(
    covariate_to_balance_1, covariate_to_balance_2
))
```

This will add a column to `df` named `treatment` with the assigned treatment.

# Python implementation

The python implementation resides in `design/`. You can perform a shallow clone of just this directory as follows:

```{bash}
mkdir design-local
cd design-local
git init
git remote add origin -f https://github.com/ddimmery/softblock.git
```

Add the `design` directory to `.git/info/sparse-checkout`

then `git pull origin master` will clone the contents of `design/`

and may be used as follows:

```{python}
import design
import numpy as np

X = np.random.rand(100,5) # numeric numpy matrix on which to balance
softblock = design.SoftBlock()
softblock.fit(X)
A = softblock.assign(X)
```

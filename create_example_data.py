
import dgp
import design
import numpy as np
import pandas as pd

np.random.seed(123)
factory = dgp.QuickBlockFactory(N=50, K=2)
dgp = factory.create_dgp()
# softblock = design.SoftBlock()
# softblock.fit(dgp.X)
# A = softblock.assign(dgp.X)
nn = design.MatchedPair()
# nn = design.GreedyNeighbors()
nn.fit(dgp.X)
A = nn.assign(dgp.X)

df = pd.DataFrame(dgp.X)
df.columns = ["X1", "X2"]
df["A"] = A
df["Y"] = dgp.Y(A)

#L = pd.DataFrame((-softblock.L).todense())
L = pd.DataFrame((nn.L).todense())

#print(L.iloc[0:4, 0:4])
#print((-softblock.L).todense()[0:4, 0:4])

df.to_csv('example_data.csv', index=False)
L.to_csv('example_laplacian.csv', index=False)

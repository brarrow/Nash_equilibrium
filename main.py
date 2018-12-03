import networkx as nx
import networkx.algorithms as alg
import matplotlib.pyplot as plt
import numpy as np

con = [[0, 0, 1, 1],[1,0,1,0],[0,0,0,1],[1,1,1,0]]
A = [[0, 0, 2, 3],[1,0,2,0],[0,0,0,1],[1,2,1,0]]
C = [[0, 0, 2, 3],[1,0,2,0],[0,0,0,1],[1,2,1,0]]
D = [[0,40,0,0],[0,0,70,0],[0,80,0,0],[0,0,0,0]]
X1 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
demand = np.nonzero(D)
demand = [(demand[0][i], demand[1][i]) for i in range(demand[0].size)]
paths = []

def curve_weight(i, j, X):
    return (A[i][j] + (X[i][j] / C[i][j]) ** 4)

def curve_grad_weight(X, alf, D):
    npA = np.array(A)
    npC = np.array(C)
    npD = np.array(D)
    npX = np.array(X)

    return npA*(npX + npD*alf) + (npX + npD*alf)**5 / (5 * npC**4)

def tpl(i,j, X):
    return (i,j,curve_weight(i,j, X))

def getGraph(X):
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    G.add_weighted_edges_from(
        [tpl(0, 2, X), tpl(0, 3, X), tpl(1, 0, X), tpl(1, 2, X), tpl(2, 3, X), tpl(3, 0, X), tpl(3, 1, X),
         tpl(3, 2, X)])
    return G

def getAlpha(X, D):
    a = 0
    b = 1
    eps = 0.001
    k = 0.001

    while b-a > eps:
        x1 = (a + b) / 2 - k * (b - a) / 2;
        x2 = (a + b) / 2 + k * (b - a) / 2;
        



G = getGraph(X1)

for curve in demand:
    paths.append(list(nx.shortest_simple_paths(G,*curve))[0])

for path in paths:
    for i in range(len(path)-1):
        ival = path[i]
        inval = path[i+1]
        X1[ival][inval] += D[path[0]][path[len(path) - 1]]

while True:
    paths.clear()
    T = getGraph(X1)
    Y1 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    for curve in demand:
        paths.append(list(nx.shortest_simple_paths(T, *curve))[0])

    for path in paths:
        for i in range(len(path) - 1):
            ival = path[i]
            inval = path[i + 1]
            Y1[ival][inval] += D[path[0]][path[len(path) - 1]]
    d = np.array(Y1) - np.array(X1)



plt.show()

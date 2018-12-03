import networkx as nx
import networkx.algorithms as alg
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

con = np.array([[0, 0, 1, 1],[1,0,1,0],[0,0,0,1],[1,1,1,0]])
A = np.array([[0, 0, 2, 3],[1,0,2,0],[0,0,0,1],[1,2,1,0]])
C = np.array([[0, 0, 2, 3],[1,0,2,0],[0,0,0,1],[1,2,1,0]])
D = np.array([[0,40,0,0],[0,0,70,0],[0,80,0,0],[0,0,0,0]])
X1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
demand = np.nonzero(D)
demand = [(demand[0][i], demand[1][i]) for i in range(demand[0].size)]
paths = []
eps = 0.0000001

def curve_weight(i, j, X):
    return (A[i][j] + (X[i][j] / C[i][j]) ** 4)

def weight(X):
    return A + (X / C)**4

def grad_weight(X, alf, D):
    ax = A * X
    dalf = D*alf

    return A*(X + D*alf) + (X + D*alf)**5 / (5 * C**4)

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
        Y1 = grad_weight(X, x1, D)
        Y2 = grad_weight(X, x2, D)
        y1s = np.nansum(Y1)
        y2s = np.nansum(Y2)

        if(y1s <= y2s):
            b = x2
        else:
            a = x1
    return (a+b)/2



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
    tw = weight(X1)
    T = getGraph(tw)
    Y1 = X1.copy()

    for curve in demand:
        paths.append(list(nx.shortest_simple_paths(T, *curve))[0])

    for path in paths:
        for i in range(len(path) - 1):
            ival = path[i]
            inval = path[i + 1]
            Y1[ival][inval] += D[path[0]][path[len(path) - 1]]
    d = Y1 - X1
    alf = getAlpha(X1, d)
    X2 = X1 + alf * d
    if norm(X2 - X1) / norm(X1) < eps:
        break
    else:
        X1 = X2
print(X2)

plt.show()

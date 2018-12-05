import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import inf
from numpy.linalg import norm
import matplotlib as mpl

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

con = np.array([[0, 0, 1, 1], [1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0]])
A = np.array([[inf, inf, 2, 3], [1, inf, 2, inf], [inf, inf, inf, 1], [1, 2, 1, inf]])
C = np.array([[0, 0, 2, 3], [1, 0, 2, 0], [0, 0, 0, 1], [1, 2, 1, 0]])
D = np.array([[0, 40, 0, 0], [0, 0, 70, 0], [0, 80, 0, 0], [0, 0, 0, 0]])
demand = np.nonzero(D)
demand = [(demand[0][i], demand[1][i]) for i in range(demand[0].size)]
paths = []
eps = 0.00001


def curve_weight(i, j, X):
    return A[i][j] + (X[i][j] / C[i][j]) ** 4


def weight(X):
    return A + (X / C) ** 4


def grad_weight(X, alf, D):
    return A * (X + D * alf) + (X + D * alf) ** 5 / (5 * (C ** 4))


def tpl(i, j, X):
    return i, j, curve_weight(i, j, X)


def get_graph(X):
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    G.add_weighted_edges_from(
        [tpl(0, 2, X), tpl(0, 3, X), tpl(1, 0, X), tpl(1, 2, X), tpl(2, 3, X), tpl(3, 0, X), tpl(3, 1, X),
         tpl(3, 2, X)])
    return G


def get_alpha_dummy(X, D):
    alphas = np.linspace(0, 1, 10000)
    min = np.inf
    alf_min = 0
    for alf in alphas:
        buf = np.nansum(grad_weight(X, alf, D))
        if buf < min:
            min = buf
            alf_min = alf
    return alf_min


def get_alpha(X, D):
    a = 0
    b = 1
    eps = 0.00001
    k = 0.001
    while b - a > eps:
        x1 = (a + b) / 2 - k * (b - a) / 2;
        x2 = (a + b) / 2 + k * (b - a) / 2;
        Y1 = grad_weight(X, x1, D)
        Y2 = grad_weight(X, x2, D)
        y1s = np.nansum(Y1)
        y2s = np.nansum(Y2)

        if (y1s <= y2s):
            b = x2
        else:
            a = x1
    return (a + b) / 2


def draw_graph(G):
    pos = nx.layout.circular_layout(G)
    posl = nx.layout.circular_layout(G, scale=1.1)

    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=10, edge_color=edge_colors,
                                   edge_cmap=plt.cm.Blues, width=2)
    labels = nx.draw_networkx_labels(G, posl)

    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


X1 = weight(np.zeros((4, 4)))

G = get_graph(X1)

for curve in demand:
    paths.append(list(nx.shortest_simple_paths(G, *curve, weight='weight'))[::-1][0])

for path in paths:
    for i in range(len(path) - 1):
        ival = path[i]
        ivaln = path[i + 1]
        X1[ival][ivaln] += D[path[0]][path[len(path) - 1]]
iter = 1
while True:
    paths.clear()
    np.nan_to_num(X1)
    tw = weight(X1)
    tw = np.nan_to_num(tw)
    T = get_graph(tw)
    Y1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    for curve in demand:
        paths.append(list(nx.shortest_simple_paths(T, *curve, weight='weight'))[0])

    for path in paths:
        for i in range(len(path) - 1):
            ival = path[i]
            ivaln = path[i + 1]
            Y1[ival][ivaln] += D[path[0]][path[len(path) - 1]]
    d = Y1 - X1
    alf = get_alpha_dummy(X1, d)
    X2 = X1 + alf * d
    X1 = np.nan_to_num(X1)
    X2 = np.nan_to_num(X2)
    print("Iter: ", iter, ", Break crit.: ", norm(X2 - X1) / norm(X1), sep="")
    iter += 1
    if norm(X2 - X1) / norm(X1) < eps:
        break
    else:
        X1 = X2
print(X2)

print("Sum t(x_res): ", np.nansum(weight(X2)))

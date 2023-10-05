import os
import copy
import random
import numpy as np
import pandas as pd

import igraph

from data_loader.dk_fullnames_to_shortnames import dk_full_to_short


def prepareMatrix(matrix):
    # Extract abbreviated names
    area_list = list(matrix.index.map(dk_full_to_short))
    # Construct matrix of relative and absolute outdegrees
    conn_matrix = np.copy(matrix)
    conn_matrix = conn_matrix / np.sum(matrix, axis=1)[:, np.newaxis]
    np.fill_diagonal(conn_matrix, 0)
    conn_matrix_abs = np.copy(matrix)
    np.fill_diagonal(conn_matrix_abs, 0)
    return area_list, conn_matrix, conn_matrix_abs


def getCommunityStructure(matrix, work_dir, infomap_path):
    """
    Determine the community structure using mapequation.
    """
    os.makedirs(work_dir, exist_ok=True)
    area_list, conn_matrix, conn_matrix_abs = prepareMatrix(matrix)

    # 1. write out graph to file for map equation
    net_fn = os.path.join(work_dir, 'Model.net')
    # nodes
    with open(net_fn, 'w') as f:
        f.write('*Vertices ' + str(len(area_list)) + '\n')
        for ii, area in enumerate(area_list):
            f.write(str(ii + 1) + ' "' + area + '"\n')
        # Determine number of vertices in the network
        k = np.where(conn_matrix != 0)[0].size
        f.write('*Arcs ' + str(k) + '\n')
        # edges
        for ii in range(len(area_list)):
            for jj in range(len(area_list)):
                if conn_matrix[ii][jj] > 0.:
                    f.write(str(jj + 1) + ' ' + str(ii + 1) +
                            ' ' + str(conn_matrix[ii][jj]) + '\n')

    # 2. Execute map equation algorithm
    infomap_exec = os.path.join(infomap_path, 'Infomap')
    ret = os.system(infomap_exec + ' --directed --tree --verbose ' +
                    net_fn + ' ' + work_dir)
    if ret != 0:
        raise OSError("Executing infomap failed. Did you install "
                      "infomap and provide the correct path by "
                      "defining the variable infomap_path?")

    mapeq_outfn = os.path.join(work_dir, 'Model.tree')
    # read tree file (see https://www.mapequation.org/infomap/#OutputTree)
    mapeq_out = pd.read_csv(mapeq_outfn, sep=' ', comment='#',
                            names=['path', 'flow', 'name', 'node_id'])
    # get top level modules as list
    map_equation = mapeq_out['path'].str.split(':').str[0]
    map_equation = map_equation.astype(int).tolist()
    # get area names corresponding to the nodes as list
    map_equation_areas = mapeq_out['name'].tolist()

    # sort map_equation lists
    index = []
    for ii in range(len(area_list)):
        index.append(map_equation_areas.index(area_list[ii]))
    map_equation = np.array(map_equation)
    membership = map_equation[index]

    # return membership info for all areas
    return membership


def plotCommunityStructure(matrix, membership, filename, edgeweight,
                           visual_style, colors=None, center_of_masses=None,
                           seed=None):
    """
    Plot community structure using igraph. Allows for custom placement and
    coloring of communities.
    """
    if colors:
        assert(np.max(membership) <= len(colors))
    if center_of_masses:
        assert(np.max(membership) <= len(center_of_masses))
    random.seed(seed)  # igraph uses random -> this fixes the seed
    area_list, conn_matrix, conn_matrix_abs = prepareMatrix(matrix)

    # Create igraph.Graph instances
    g = createGraph(conn_matrix, area_list)
    g_abs = createGraph(conn_matrix_abs, area_list)

    # Copy the graphs for further modification necessary for plotting
    gplot = g.copy()
    gplot_abs = g_abs.copy()
    gcopy = g.copy()
    gplot.delete_edges(None)
    gplot_abs.delete_edges(None)
    edges = []
    for edge in g.es():
        weight = g.es.select(_source=edge.tuple[0],
                             _target=edge.tuple[1])['weight']
        same_cluster = membership[edge.tuple[0]] == membership[edge.tuple[1]]
        if not same_cluster or len(weight) == 0:
            edges.append(edge)
    gcopy.delete_edges(edges)

    edges_colors = []
    # Inter-cluster connections are gray
    for edge_id, edge in enumerate(g.es()):
        relevant_edge = edge['weight'] > 0.001
        same_cluster = membership[edge.tuple[0]] == membership[edge.tuple[1]]
        if relevant_edge and not same_cluster:
            gplot.add_edge(edge.tuple[0], edge.tuple[1], weight=edge['weight'])
            gplot_abs.add_edge(edge.tuple[0], edge.tuple[1],
                               weight=g_abs.es()[edge_id]['weight'])
            edges_colors.append("gray")
    # Intra-cluster connections are black (separate loop to order edges)
    for edge_id, edge in enumerate(g.es()):
        relevant_edge = edge['weight'] > 0.001
        same_cluster = membership[edge.tuple[0]] == membership[edge.tuple[1]]
        if relevant_edge and same_cluster:
            gplot.add_edge(edge.tuple[0], edge.tuple[1], weight=edge['weight'])
            gplot_abs.add_edge(edge.tuple[0], edge.tuple[1],
                               weight=g_abs.es()[edge_id]['weight'])
            edges_colors.append("black")

    # Inside cluster, distribute areas using a force-directed algorithm
    # by Kamada and Kawai, 1989
    layout_params = {'maxy': list(range(len(area_list)))}
    layout = gcopy.layout("kk", **layout_params)
    # For better visibility, place clusters at defined positions
    if center_of_masses is not None:
        coords = np.array(copy.copy(layout.coords))
        for ii in range(np.max(membership)):
            coo = coords[np.where(membership == ii + 1)]
            com = np.mean(coo, axis=0)
            coo = np.array(coo) - (com - center_of_masses[ii])
            coords[np.where(membership == ii + 1)] = coo
        visual_style["layout"] = list(coords)
    # Define layout parameters
    gplot.es["color"] = edges_colors
    visual_style["edge_color"] = edges_colors
    visual_style["vertex_label"] = gplot.vs["name"]

    weights_transformed = np.log(gplot_abs.es['weight']) * edgeweight
    visual_style["edge_width"] = weights_transformed

    for vertex in gplot.vs():
        vertex["label"] = vertex.index
    if colors is not None and membership is not None:
        for vertex in gplot.vs():
            vertex["color"] = colors[membership[vertex.index] - 1]
        visual_style["vertex_color"] = gplot.vs["color"]

    # use igraph's plot function to finally plot the graph and save to file
    igraph.plot(gplot, filename, **visual_style)

    return 0


def createGraph(matrix, area_list):
    """
    Create igraph.Graph instance from a given connectivity matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Connectivity matrix
    area_list: list or numpy.ndarray
        List of areas

    Returns
    -------
    g : igraph.Graph
    """
    g = igraph.Graph(directed=True)
    g.add_vertices(area_list)

    for ii in range(len(area_list)):
        for jj in range(len(area_list)):
            if matrix[ii][jj] != 0:
                g.add_edge(jj, ii, weight=matrix[ii][jj])
    return g

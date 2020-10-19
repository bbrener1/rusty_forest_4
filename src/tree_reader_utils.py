
import numpy as np
from scipy.spatial.distance import pdist,cdist,squareform

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

def numpy_mad(mtx):
    medians = []
    for column in mtx.T:
        medians.append(np.median(column[column != 0]))
    median_distances = np.abs(
        mtx - np.tile(np.array(medians), (mtx.shape[0], 1)))
    mads = []
    for (i, column) in enumerate(median_distances.T):
        mads.append(np.median(column[mtx[:, i] != 0]))
    return np.array(mads)


def ssme(mtx, axis=None):
    medians = np.median(mtx, axis=0)
    median_distances = mtx - np.tile(np.array(medians), (mtx.shape[0], 1))
    ssme = np.sum(np.power(median_distances, 2), axis=axis)
    return ssme



def hacked_louvain(knn, resolution=1):
    import louvain
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    g = ig.Graph()
    g.add_vertices(knn.shape[0])  # this adds adjacency.shape[0] vertices
    edges = [(s, t) for s in range(knn.shape[0]) for t in knn[s]]

    g.add_edges(edges)

    if g.vcount() != knn.shape[0]:
        logg.warning(
            f'The constructed graph has only {g.vcount()} nodes. '
            'Your adjacency matrix contained redundant nodes.'
        )

    print("Searching for partition")
    part = louvain.find_partition(
        g, partition_type=louvain.RBConfigurationVertexPartition, resolution_parameter=resolution)
    clustering = np.zeros(knn.shape[0], dtype=int)
    for i in range(len(part)):
        clustering[part[i]] = i
    print("Louvain: {}".format(clustering.shape))
    return clustering


# def embedded_hdbscan(coordinates):
#
#     clustering_model = HDBSCAN(min_cluster_size=50)
#     clusters = clustering_model.fit_predict(coordinates)
#     return clusters
#
# def sample_hdbscan(nodes,samples):
#
#     node_encoding = node_sample_encoding(nodes,samples)
#     embedding_model = PCA(n_components=100)
#     pre_computed_embedded = embedding_model.fit_transform(node_encoding.T)
#     print("Sample HDBscan Encoding: {}".format(pre_computed_embedded.shape))
# #     pre_computed_distance = coocurrence_distance(node_encoding)
#     pre_computed_distance = scipy.spatial.distance.squareform(pdist(pre_computed_embedded,metric='correlation'))
#     print("Sample HDBscan Distance Matrix: {}".format(pre_computed_distance.shape))
# #     pre_computed_distance[pre_computed_distance == 0] += .000001
#     pre_computed_distance[np.isnan(pre_computed_distance)] = 10000000
#     clustering_model = HDBSCAN(min_samples=3,metric='precomputed')
#     clusters = clustering_model.fit_predict(pre_computed_distance)
#
#     return clusters




def sample_agglomerative(nodes, samples, n_clusters):

    node_encoding = node_sample_encoding(nodes, samples)

    pre_computed_distance = pdist(node_encoding.T, metric='cosine')

    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters, affinity='precomputed')

    clusters = clustering_model.fit_predict(
        scipy.spatial.distance.squareform(pre_computed_distance))

#     clusters = clustering_model.fit_predict(node_encoding)

    return clusters


def stack_dictionaries(dictionaries):
    stacked = {}
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if key not in stacked:
                stacked[key] = []
            stacked[key].append(value)
    return stacked


def partition_mutual_information(p1, p2):
    p1 = p1.astype(dtype=float)
    p2 = p2.astype(dtype=float)
    population = p1.shape[1]
    intersections = np.dot(p1, p2.T)
    partition_size_products = np.outer(np.sum(p1, axis=1), np.sum(p2, axis=1))
    log_term = np.log(intersections) - \
        np.log(partition_size_products) + np.log(population)
    log_term[np.logical_not(np.isfinite(log_term))] = 0
    mutual_information_matrix = (intersections / population) * log_term
    return mutual_information_matrix



def count_list_elements(elements):
    dict = {}
    for element in elements:
        if element not in dict:
            dict[element] = 0
        dict[element] += 1
    return dict


def triangulate_knn(elements, k):

    distances = {}

    anchor = 0


def generate_feature_value_html(features, values, normalization=None, cmap=None):

    if not isinstance(cmap, mpl.colors.Colormap):
        from matplotlib.cm import get_cmap
        try:
            cmap = get_cmap(cmap)
        except:
            cmap = get_cmap('viridis')
    # if normalization is None:
    #     from matplotlib.colors import SymLogNorm, DivergingNorm
    #     normalization = DivergingNorm(0)
        # normalization = SymLogNorm(linthresh=.05)

    html_elements = [
        # '<table width="100%">',
        '<table>',
        "<style>", "th,td {padding:5px;border-bottom:1px solid #ddd;}", "</style>",
        "<tr>",
        "<th>", "Features", "</th>",
        "<th>", "Values", "</th>",
        "</tr>",
    ]
    for feature, value in zip(features, values):
        value_color_tag = ""
        # if normalization is not None:
        #     normed_value = normalization(value)
        #     r,g,b,a = cmap(normed_value)
        #     r,g,b,a = r*100,g*100,b*100,a*100
        # value_color_tag = f'style="background-color:rgba({r}%,{g}%,{b}%,50%);"'
        # value_color_tag = f'style="background-image:linear-gradient(to right,rgba({r}%,{g}%,{b}%,0%),rgba({r}%,{g}%,{b}%,50%));"'
        feature_elements = f"""
            <tr>
                <td>{feature}</td>
                <td {value_color_tag}>{value}</td>
            </td>
        """
        html_elements.append(feature_elements)

    html_elements.append("</table>")
    return "".join(html_elements)


# def generate_local_correlation_table(f1,f2,local,global):
#
#     html_elements = [
#         # '<table width="100%">',
#         '<table>',
#         "<style>", "th,td {padding:5px;border-bottom:1px solid #ddd;}", "</style>",
#         "<tr>",
#         "<th>", "Features", "</th>",
#         "<th>", "Values", "</th>",
#         "</tr>",
#     ]


def js_wrap(name, content):
    return f"<script> let {name} = {content};</script>"


def fast_knn(elements, k, neighborhood_fraction=.01, metric='euclidean'):

    # Finds the indices of k nearest neighbors for each sample in a matrix,
    # using any of the standard scipy distance metrics.

    nearest_neighbors = np.zeros((elements.shape[0], k), dtype=int)
    complete = np.zeros(elements.shape[0], dtype=bool)

    neighborhood_size = max(
        k * 3, int(elements.shape[0] * neighborhood_fraction))
    anchor_loops = 0

    while np.sum(complete) < complete.shape[0]:

        anchor_loops += 1

        available = np.arange(complete.shape[0])[~complete]
        np.random.shuffle(available)
        anchors = available[:int(complete.shape[0] / neighborhood_size) * 3]

        for anchor in anchors:
            print(f"Complete:{np.sum(complete)}\r", end='')

            anchor_distances = cdist(elements[anchor].reshape(
                1, -1), elements, metric=metric)[0]

            neighborhood = np.argpartition(anchor_distances, neighborhood_size)[
                :neighborhood_size]
            anchor_local = np.where(neighborhood == anchor)[0]

            local_distances = squareform(
                pdist(elements[neighborhood], metric=metric))

            anchor_to_worst = np.max(local_distances[anchor_local])

            for i, sample in enumerate(neighborhood):
                if not complete[sample]:

                    # First select the indices in the neighborhood that are knn
                    best_neighbors_local = np.argpartition(
                        local_distances[i], k + 1)

                    # Next find the worst neighbor among the knn observed
                    best_worst_local = best_neighbors_local[np.argmax(
                        local_distances[i][best_neighbors_local[:k + 1]])]
                    # And store the worst distance among the local knn
                    best_worst_distance = local_distances[i, best_worst_local]
                    # Find the distance of the anchor to the central element
                    anchor_distance = local_distances[anchor_local, i]

                    # By the triangle inequality the closest any element outside the neighborhood
                    # can be to element we are examining is the criterion distance:
                    criterion_distance = anchor_to_worst - anchor_distance

#                     if sample == 0:
#                         print(f"ld:{local_distances[i][best_neighbors_local[:k]]}")
#                         print(f"bwd:{best_worst_distance}")
#                         print(f"cd:{criterion_distance}")

                    # Therefore if the criterion distance is greater than the best worst distance, the local knn
                    # is also the best global knn

                    if best_worst_distance >= criterion_distance:
                        continue
                    else:
                        # Before we conclude we must exclude the sample itself from its
                        # k nearest neighbors
                        best_neighbors_local = [
                            bn for bn in best_neighbors_local[:k + 1] if bn != i]
                        # Finally translate the local best knn to the global indices
                        best_neighbors = neighborhood[best_neighbors_local]

                        nearest_neighbors[sample] = best_neighbors
                        complete[sample] = True
    print("\n")

    return nearest_neighbors


def double_fast_knn(elements1, elements2, k, neighborhood_fraction=.01, metric='cosine'):

    if elements1.shape != elements2.shape:
        raise Exception("Average metric knn inputs must be same size")

    nearest_neighbors = np.zeros((elements1.shape[0], k), dtype=int)
    complete = np.zeros(elements1.shape[0], dtype=bool)

    neighborhood_size = max(
        k * 3, int(elements1.shape[0] * neighborhood_fraction))
    anchor_loops = 0
    # failed_counter = 0

    while np.sum(complete) < complete.shape[0]:

        anchor_loops += 1

        available = np.arange(complete.shape[0])[~complete]
        np.random.shuffle(available)
        anchors = available[:int(complete.shape[0] / neighborhood_size) * 3]

        for anchor in anchors:
            print(f"Complete:{np.sum(complete)}\r", end='')

            ad_1 = cdist(elements1[anchor].reshape(
                1, -1), elements1, metric=metric)[0]
            ad_2 = cdist(elements2[anchor].reshape(
                1, -1), elements2, metric=metric)[0]
            anchor_distances = (ad_1 + ad_2) / 2

    #         print(f"anchor:{anchor}")

            neighborhood = np.argpartition(anchor_distances, neighborhood_size)[
                :neighborhood_size]
            anchor_local = np.where(neighborhood == anchor)[0]

    #         print(neighborhood)

            ld_1 = squareform(pdist(elements1[neighborhood], metric=metric))
            ld_2 = squareform(pdist(elements2[neighborhood], metric=metric))
            local_distances = (ld_1 + ld_2) / 2

            anchor_to_worst = np.max(local_distances[anchor_local])

            for i, sample in enumerate(neighborhood):
                if not complete[sample]:

                    # First select the indices in the neighborhood that are knn
                    best_neighbors_local = np.argpartition(
                        local_distances[i], k + 1)

                    # Next find the worst neighbor among the knn observed
                    best_worst_local = best_neighbors_local[np.argmax(
                        local_distances[i][best_neighbors_local[:k + 1]])]
                    # And store the worst distance among the local knn
                    best_worst_distance = local_distances[i, best_worst_local]
                    # Find the distance of the anchor to the central element
                    anchor_distance = local_distances[anchor_local, i]

                    # By the triangle inequality the closest any element outside the neighborhood
                    # can be to element we are examining is the criterion distance:
                    criterion_distance = anchor_to_worst - anchor_distance

#                     if sample == 0:
#                         print(f"ld:{local_distances[i][best_neighbors_local[:k]]}")
#                         print(f"bwd:{best_worst_distance}")
#                         print(f"cd:{criterion_distance}")

                    # Therefore if the criterion distance is greater than the best worst distance, the local knn
                    # is also the best global knn

                    if best_worst_distance >= criterion_distance:
                        continue
                    else:
                        # Before we conclude we must exclude the sample itself from its
                        # k nearest neighbors
                        best_neighbors_local = [
                            bn for bn in best_neighbors_local[:k + 1] if bn != i]
                        # Finally translate the local best knn to the global indices
                        best_neighbors = neighborhood[best_neighbors_local]

                        nearest_neighbors[sample] = best_neighbors
                        complete[sample] = True
    print("\n")

    return nearest_neighbors

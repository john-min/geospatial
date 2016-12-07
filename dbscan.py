import itertools

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import DistanceMetric, NearestNeighbors

from operator import itemgetter
from utils import haversine

'''
DISTANCE COMPUTATION
'''

def dist2NN(X):
    '''
    Distance to Nearest Neighbor of precomputed distance matrix X
    '''
    return [sorted(row)[1] for row in X]

def pairwise_distance(X):
    df = pd.DataFrame(X).drop_duplicates()
    return df, squareform(pdist(df, lambda u,v: haversine(u,v)))

def count_elems_lt(array, threshold):
    return len([elem for elem in array if elem <= threshold])

def count_neighbors(X, epsilon):
    '''
    Count neighbors lying in point's epsilon-neighborhood
    '''
    return [count_elems_lt(row, epsilon) for row in X]

def epsilon_eda(X, n_bins=1000):

    dist2nn = dist2NN(X)
    fig, ax = plt.subplots()
    ax.hist(dist2nn, bins=n_bins)

    fig, ax = plt.subplots()
    ax.boxplot(dist2nn)

def neighborhood_eda(X, eps, n_bins=25):
    neighbors_eps = count_neighbors(X_pairwise, eps)

    fig, ax = plt.subplots()
    ax.hist(neighbors_eps, bins=n_bins)
    fig.show()

    fig, ax = plt.subplots()
    ax.boxplot(neighbors_eps)

def dbscan_grid_search(X, eps_search=[0.005, 0.01, 0.015, 0.02], min_samples_search=[10,25,50,100,150]):
    results = []
    for e,n in itertools.product(eps_search, min_samples_search):
        try:
            db = DBSCAN(eps=e, min_samples=n, metric='haversine').fit(X)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            silhouette_coef = metrics.silhouette_score(X, labels)
            results.append((e,n,n_clusters_,silhouette_coef, db))
        except:
            #results.append((e,n,n_clusters_,None))
            results.append((e,n,n_clusters_,0,0)) #put 0 instead of None for result comparison
    return results

def find_best_result(results, idx=3):
    return max(results, key=itemgetter(idx))


def fit_dbscan(X, eps_search=[0.005, 0.01, 0.015, 0.02], min_samples_search=[10,25,50,100,150]):
    dbscan_results = dbscan_grid_search(X, eps_search, min_samples_search)
    # Print results of DBSCAN grid search
    results_df = pd.DataFrame(dbscan_results)
    results_df.columns = ['epsilon', 'min_pts', 'n_clusters', 'silhouette_score', 'db_obj']
    print (('epsilon', 'min_pts', 'n_clusters', 'silhouette_score'))
    for idx, row in results_df.iterrows():
        print ((row['epsilon'], row['min_pts'], row['n_clusters'], row['silhouette_score']))

    # Plot DBSCAN clustering results
    eps, min_pts, n_clusters_, silhouette_score, db = find_best_result(dbscan_results)
    #db = dbscan_results[8][4]
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    print ('epsilon: %f, min_pts: %d' % (eps,min_pts))
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=7)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=5)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return

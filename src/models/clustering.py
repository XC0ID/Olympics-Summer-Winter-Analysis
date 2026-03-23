from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

def build_kmeans(n_clusters=5, random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state,
                          n_init=10, max_iter=300)),
    ])

def find_optimal_k(X, k_range=range(2, 11), random_state=42):
    Xs = StandardScaler().fit_transform(X)
    results = {}
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(Xs)
        results[k] = {
            "inertia":    round(km.inertia_, 2),
            "silhouette": round(silhouette_score(Xs, labels), 4),
        }
    return results
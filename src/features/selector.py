from sklearn.feature_selection import SelectKBest, f_regression

def select_top_k(X, y, feature_names, k=10):
    selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
    selector.fit(X, y)
    mask     = selector.get_support()
    selected = [feature_names[i] for i, m in enumerate(mask) if m]
    print(f"Selected features: {selected}")
    return selector.transform(X), selected, selector 
import numpy as np

class DecisionTreeCART:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        p = counts / m
        gini_index = 1 - np.sum(p ** 2)
        return gini_index
    
    def _best_split(self, X, y):
        min_gini = float("inf")
        best_split = None
        m, n = X.shape

        for feature_idx in range(n):
            for threshold in np.unique(X[:, feature_idx]):
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                gini_split = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / m
                if gini_split < min_gini:
                    min_gini = gini_split
                    best_split = {
                        "feature_idx": feature_idx,
                        "threshold": threshold,
                        "gini": min_gini,
                        "left_mask": left_mask,
                        "right_mask": right_mask
                    }

        return best_split

    def _build_tree(self, X, y, depth=0):
        m, n = X.shape
        num_classes = len(np.unique(y))

        if depth >= self.max_depth or num_classes == 1 or m < self.min_samples_split:
            most_common_class = np.bincount(y).argmax()
            return most_common_class
        
        split = self._best_split(X, y)
        if not split:
            most_common_class = np.bincount(y).argmax()
            return most_common_class
        
        left = self._build_tree(X[split["left_mask"]], y[split["left_mask"]], depth + 1)
        right = self._build_tree(X[split["right_mask"]], y[split["right_mask"]], depth + 1)

        return {
            "feature_idx": split["feature_idx"],
            "threshold": split["threshold"],
            "left": left,
            "right": right
        }
    
    def fit(self, X, y):
        self.tree = self._build_tree(X.values, y.values)

    def _predict(self, x, tree):
        if isinstance(tree, dict):
            feature_idx = tree["feature_idx"]
            threshold = tree["threshold"]
            if x[feature_idx] <= threshold:
                return self._predict(x, tree["left"])
            else:
                return self._predict(x, tree["right"])
        else:
            return tree
        
    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X.values])
    
    def get_params(self, deep=True):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split}
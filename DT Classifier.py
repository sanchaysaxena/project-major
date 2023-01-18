from loading_data import X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc,X_train_chroma, X_test_chroma, y_train_chroma, y_test_chroma, X_train_tonnetz, X_test_tonnetz, y_train_tonnetz, y_test_tonnetz
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _is_finished(self, depth):
        if (depth >= self.max_depth
                or self.n_class_labels == 1
                or self.n_samples < self.min_samples_split):
            return True
        return False

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {'score': - 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)



def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# X_train = np.array(X_train)[indices.astype(int)]
# X_test = np.array(X_test)[indices.astype(int)]
# y_train = np.array(y_train)[indices.astype(int)]
# y_test = np.array(y_test)[indices.astype(int)]


clf = DecisionTree(max_depth=6)

clf.fit(X_train_mfcc, y_train_mfcc)
y_pred_mfcc = clf.predict(X_test_mfcc)
acc_mfcc = accuracy(y_test_mfcc, y_pred_mfcc)

clf.fit(X_train_chroma,y_train_chroma)
y_pred_chroma = clf.predict(X_test_chroma)
acc_chroma = accuracy(y_test_chroma,y_pred_chroma)

clf.fit(X_train_tonnetz,y_train_tonnetz)
y_pred_tonnetz = clf.predict(X_test_tonnetz)
acc_tonnetz = accuracy(y_test_tonnetz,y_pred_tonnetz)

print("Decision Trees Testing Accuracy using MFCC:", acc_mfcc)
print("Decision Trees Testing Accuracy using chroma:", acc_chroma)
print("Decision Trees Testing Accuracy using Tonnetz:", acc_tonnetz)


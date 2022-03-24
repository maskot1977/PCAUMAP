import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.impute import KNNImputer

class PCAUmap:
    def __init__(
        self,
        n_neighbors=15,
        use_pca=1,
        min_dist=0.1,
        n_components=2,
        random_state=None,
        transform_seed=None,
        scaler=True,
        metric="euclidean",
        augment_size = 3,
        impute_rate = 0.1,
    ):
        self.pca = PCA()
        self.umap = UMAP(
            random_state=random_state,
            transform_seed=transform_seed,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
        )
        self.use_pca = use_pca
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.data = None
        self.pca_features = None
        self.embedding = None
        self.imputer = KNNImputer()
        self.augment_size = augment_size
        self.impute_rate = impute_rate

    def fit(self, data):
        self.data = pd.DataFrame(data)
        augmented_data = self.augumentation(self.augment_size, self.impute_rate)
        
        if self.scaler is None:
            if self.use_pca is None:
                self.umap.fit(augmented_data)
                self.embedding = self.umap.transform(data)
            else:
                self.umap.fit(self.pca.fit_transform(augmented_data))
                self.pca_features = self.pca.transform(data)
                self.embedding = self.umap.transform(self.pca_features)
        else:
            if self.use_pca is None:
                self.umap.fit(self.scaler.fit_transform(augmented_data))
                self.embedding = self.umap.transform(self.scaler.transform(data))
            else:
                self.umap.fit(self.pca.fit_transform(self.scaler.fit_transform(augmented_data)))
                self.pca_features = self.pca.transform(self.scaler.transform(data))
                self.embedding = self.umap.transform(self.pca_features)
        return self

    def transform(self, data):
        self.data = pd.DataFrame(data)
        if self.scaler is None:
            if self.pca is None:
                self.embedding = self.umap.transform(data)
                return self.embedding
            else:
                self.pca_features = self.pca.transform(data)
                self.embedding = self.umap.transform(self.pca_features)
                return self.embedding
        else:
            if self.pca is None:
                self.embedding = self.umap.transform(self.scaler.transform(data))
                return self.embedding
            else:
                self.pca_features = self.pca.transform(self.scaler.transform(data))
                self.embedding = self.umap.transform(self.pca_features)
                return self.embedding

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, embedded):
        if self.scaler is None:
            if self.pca is None:
                return self.umap.inverse_transform(embedded)
            else:
                return self.pca.inverse_transform(self.umap.inverse_transform(embedded))
        else:
            if self.pca is None:
                return self.scaler.inverse_transform(
                    self.umap.inverse_transform(embedded)
                )
            else:
                return self.scaler.inverse_transform(
                    self.pca.inverse_transform(self.umap.inverse_transform(embedded))
                )

    def pca_summary(self, c=None):
        plt.figure(figsize=(6, 6))
        if c is None:
            plt.scatter(self.pca_features[:, 0], self.pca_features[:, 1], alpha=0.5)
        else:
            plt.scatter(
                self.pca_features[:, 0], self.pca_features[:, 1], alpha=0.5, c=c
            )
        plt.xlabel("PC1 ({}%)".format(int(self.pca.explained_variance_ratio_[0] * 100)))
        plt.ylabel("PC2 ({}%)".format(int(self.pca.explained_variance_ratio_[1] * 100)))
        plt.grid()
        plt.show()
        plt.figure(figsize=(6, 6))
        plt.scatter(self.pca.components_[0], self.pca.components_[1], alpha=0.5)
        plt.xlabel("loading 1")
        plt.ylabel("loading 2")
        plt.grid()
        plt.show()
        plt.figure(figsize=(6, 6))
        plt.plot([0] + list(np.cumsum(self.pca.explained_variance_ratio_)), "-o")
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative contribution ratio")
        plt.grid()
        plt.show()

    def map_predicted_values(
        self,
        model,
        c=None,
        alpha=0.5,
        edgecolors="k",
        figsize=(8, 6),
        h=0.2,
        cm=plt.cm.jet,
    ):

        x_min = self.embedding[:, 0].min() - 0.5
        x_max = self.embedding[:, 0].max() + 0.5
        y_min = self.embedding[:, 1].min() - 0.5
        y_max = self.embedding[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        plt.figure(figsize=figsize)
        if hasattr(model, "predict_proba"):
            Z = model.predict_proba(
                self.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
            )[:, 1]
        elif hasattr(model, "decision_function"):
            Z = model.decision_function(
                self.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
            )
        else:
            Z = model.predict(self.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))

        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=alpha, cmap=cm)
        plt.colorbar()
        if c is None:
            plt.scatter(
                self.embedding[:, 0],
                self.embedding[:, 1],
                alpha=alpha,
                edgecolors=edgecolors,
            )
        else:
            plt.scatter(
                self.embedding[:, 0],
                self.embedding[:, 1],
                alpha=alpha,
                c=c,
                edgecolors=edgecolors,
            )
        plt.grid()
        plt.show()
        
    def augumentation(self, augment_size, rate):
        augmented_data = pd.concat([self.data] * augment_size).values
        augmented_data = fill_randomly(augmented_data, np.nan, rate)
        augmented_data = pd.DataFrame(self.imputer.fit_transform(augmented_data))
        augmented_data = pd.concat([self.data, augmented_data])
        return augmented_data
    
def fill_randomly(X, value, rate):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.random.rand() < rate:
                X[np.ix_([i], [j])] = value
    return X

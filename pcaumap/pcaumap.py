from umap import UMAP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAUmap:
    def __init__(self, use_pca=1.0, random_state=53, transform_seed=53, scaler=True):
        self.pca = PCA()
        self.umap = UMAP(random_state=random_state, transform_seed=transform_seed)
        self.use_pca = use_pca
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.data = None
        self.pca_features = None
        self.embedding = None

    def fit(self, data):
        self.data = pd.DataFrame(data)
        if self.scaler is None:
            if self.use_pca is None:
                self.embedding = self.umap.fit_transform(data)
            else:
                self.pca_features = self.pca.fit_transform(data)
                self.embedding = self.umap.fit_transform(self.pca_features)
        else:
            if self.use_pca is None:
                self.embedding = self.umap.fit_transform(self.scaler.fit_tranform(data))
            else:
                self.pca_features = self.pca.fit_transform(self.scaler.fit_transform(data))
                self.embedding = self.umap.fit_transform(self.pca_features)

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
                return self.scaler.inverse_transform(self.umap.inverse_transform(embedded))
            else:
                return self.scaler.inverse_transform(self.pca.inverse_transform(self.umap.inverse_transform(embedded)))
            
    def pca_summary(self, c=None):
        if c is None:
            plt.scatter(self.pca_features[:, 0], self.pca_features[:, 1], alpha=0.5)
        else:
            plt.scatter(self.pca_features[:, 0], self.pca_features[:, 1], alpha=0.5, c=c)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid()
        plt.show()
        plt.scatter(self.pca.components_[0], self.pca.components_[1], alpha=0.5)
        plt.xlabel("loading 1")
        plt.ylabel("loading 2")
        plt.grid()
        plt.show()
        plt.plot([0] + list(np.cumsum(self.pca.explained_variance_ratio_)), "-o")
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative contribution ratio")
        plt.grid()
        plt.show()

    def map_predicted_values(self, model, c=None, alpha=0.5, edgecolors="k", figsize=(8, 6), h=0.2, cm=plt.cm.jet):

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
            plt.scatter(self.embedding[:, 0], self.embedding[:, 1], alpha=alpha, edgecolors=edgecolors)
        else:
            plt.scatter(self.embedding[:, 0], self.embedding[:, 1], alpha=alpha, c=c, edgecolors=edgecolors)
        plt.grid()
        plt.show()

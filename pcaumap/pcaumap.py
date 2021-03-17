from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PCAUmap:
    def __init__(self, use_pca=1.0, random_state=53, transform_seed=53):
        self.pca = PCA()
        self.umap = UMAP(random_state=random_state, transform_seed=transform_seed)
        self.use_pca = use_pca
        self.random_state = random_state

    def fit(self, data):
        if self.use_pca is not None:
            self.pca.fit(data)
            pca_feature = self.pca.transform(data)
            self.umap.fit(pca_feature)
        else:
            self.umap.fit(data)

    def transform(self, data):
        if self.pca is not None:
            pca_feature = self.pca.transform(data)
            return self.umap.transform(pca_feature)
        else:
            return self.umap.transform(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, embedded):
        if self.pca is not None:
            return self.pca.inverse_transform(self.umap.inverse_transform(embedded))
        else:
            return self.umap.inverse_transform(embedded)

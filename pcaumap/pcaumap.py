from umap import UMAP
import numpy as np
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

    def fit(self, data):
        self.data = data
        if self.scaler is None:
            if self.use_pca is None:
                self.umap.fit(data)
            else:
                self.pca.fit(data)
                self.umap.fit(self.pca.transform(data))
        else:
            self.scaler.fit(data)
            if self.use_pca is None:
                self.umap.fit(self.scaler.tranform(data))
            else:
                self.pca.fit(self.scaler.transform(data))
                self.pca.transform(self.scaler.transform(data))
                self.umap.fit(self.pca.transform(self.scaler.transform(data)))

    def transform(self, data):
        self.data = data
        if self.scaler is None:
            if self.pca is None:
                return self.umap.transform(data)
            else:
                self.pca_features = self.pca.transform(data)
                return self.umap.transform(self.pca_features)
        else:
            if self.pca is None:
                return self.umap.transform(self.scaler.transform(data))
            else:
                self.pca_features = self.pca.transform(self.scaler.transform(data))
                return self.umap.transform(self.pca_features)

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
            

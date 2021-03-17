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


def show_pcaumap(
    pcaumap,
    X_train,
    y_train=None,
    X_test=None,
    y_test=None,
    pca=None,
    model=None,
    h=0.5,
    cm=plt.cm.jet,
    title=None,
):
    embedding_train = pcaumap.transform(X_train)
    if X_test is not None:
        embedding_test = pcaumap.transform(X_test)

    if X_test is not None:
        x_min = min(embedding_train[:, 0].min() - 0.5, embedding_test[:, 0].min() - 0.5)
        x_max = max(embedding_train[:, 0].max() + 0.5, embedding_test[:, 0].max() + 0.5)
        y_min = min(embedding_train[:, 1].min() - 0.5, embedding_test[:, 1].min() - 0.5)
        y_max = max(embedding_train[:, 1].max() + 0.5, embedding_test[:, 1].max() + 0.5)
    else:
        x_min = embedding_train[:, 0].min() - 0.5
        x_max = embedding_train[:, 0].max() + 0.5
        y_min = embedding_train[:, 1].min() - 0.5
        y_max = embedding_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    plt.figure(figsize=(8, 6))
    if title is not None:
        plt.title(title)

    if model is not None:
        if hasattr(model, "predict_proba"):
            Z = model.predict_proba(
                pcaumap.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
            )[:, 1]
        elif hasattr(model, "decision_function"):
            Z = model.decision_function(
                pcaumap.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
            )
        else:
            Z = model.predict(pcaumap.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=cm)
        plt.colorbar()

        plt.scatter(
            embedding_train[:, 0],
            embedding_train[:, 1],
            label="train",
            facecolors="none",
            edgecolors="k",
            alpha=0.5,
        )
        if X_test is not None:
            plt.scatter(
                embedding_test[:, 0],
                embedding_test[:, 1],
                label="test",
                facecolors="none",
                edgecolors="r",
                alpha=0.5,
            )
    else:
        if y_train is not None:
            plt.scatter(
                embedding_train[:, 0],
                embedding_train[:, 1],
                edgecolors="k",
                c=y_train,
                alpha=0.5,
            )
        else:
            plt.scatter(
                embedding_train[:, 0], embedding_train[:, 1], edgecolors="k", alpha=0.5
            )
        if X_test is not None:
            if y_train is not None:
                plt.scatter(
                    embedding_test[:, 0],
                    embedding_test[:, 1],
                    edgecolors="r",
                    c=y_test,
                    alpha=0.5,
                )
            else:
                plt.scatter(
                    embedding_test[:, 0],
                    embedding_test[:, 1],
                    edgecolors="r",
                    alpha=0.5,
                )
        if y_train is not None:
            plt.colorbar()

    plt.show()


def pca_summary(
    pca,
    X_train,
    y_train=None,
    X_test=None,
    y_test=None,
    loading_color=None,
    text_limit=100,
):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6 * 3, 6),)

    pca_feature_train = pca.transform(X_train)
    if y_train is not None:
        axes[0].scatter(
            pca_feature_train[:, 0],
            pca_feature_train[:, 1],
            alpha=0.5,
            edgecolors="k",
            c=y_train,
        )
    else:
        axes[0].scatter(
            pca_feature_train[:, 0], pca_feature_train[:, 1], alpha=0.5, edgecolors="k"
        )

    if X_test is not None:
        pca_feature_test = pca.transform(X_test)
        if y_test is not None:
            axes[0].scatter(
                pca_feature_test[:, 0],
                pca_feature_test[:, 1],
                alpha=0.5,
                edgecolors="r",
                c=y_test,
            )
        else:
            axes[0].scatter(pca_feature_test[:, 0], pca_feature_test[:, 1], alpha=0.5)

    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid()

    if loading_color is None:
        axes[1].scatter(pca.components_[0], pca.components_[1], edgecolors="k")
    else:
        axes[1].scatter(pca.components_[0], pca.components_[1], edgecolors="k", c=loading_color)

    if len(pca.components_[0]) < text_limit:
        for x, y, name in zip(pca.components_[0], pca.components_[1], X_train.columns):
            axes[1].text(x, y, name)

    axes[1].set_xlabel("PC1 loading")
    axes[1].set_ylabel("PC2 loading")
    axes[1].grid()

    axes[2].plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
    axes[2].set_xlabel("Number of principal components")
    axes[2].set_ylabel("Cumulative contribution ratio")
    axes[2].grid()
    plt.show()

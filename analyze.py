import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles
from unet import *


class ManifoldInteractionAnalyzer:
    """
    Analyze latent manifold geometry of:
        - CT-only model
        - PET-only model
        - PET+CT model

    Features should already be extracted:

        ct_feats      : (N,D)
        pet_feats     : (N,D)
        fusion_feats  : (N,D)

    """

    def __init__(
        self,
        ct_feats,
        pet_feats,
        fusion_feats,
        n_neighbors=30,
        tangent_dim=10,
    ):
        self.ct = ct_feats
        self.pet = pet_feats
        self.fusion = fusion_feats

        self.n_neighbors = n_neighbors
        self.tangent_dim = tangent_dim

    # --------------------------------------------------
    # Tangent estimation
    # --------------------------------------------------

    def estimate_tangent_space(self, features, idx):
        """
        Local PCA around sample idx.
        """

        nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors
        ).fit(features)

        _, indices = nbrs.kneighbors(
            features[idx:idx+1]
        )

        local_points = features[
            indices[0]
        ]

        centered = (
            local_points -
            local_points.mean(axis=0)
        )

        pca = PCA(
            n_components=self.tangent_dim
        )

        pca.fit(centered)

        return pca.components_.T

    # --------------------------------------------------
    # Grassmann distance
    # --------------------------------------------------

    def grassmann_distance(
        self,
        basis_A,
        basis_B
    ):
        """
        Projection Frobenius distance.
        """

        PA = basis_A @ basis_A.T
        PB = basis_B @ basis_B.T

        return np.linalg.norm(
            PA - PB,
            ord="fro"
        )

    # --------------------------------------------------
    # Principal angles
    # --------------------------------------------------

    def principal_angles(
        self,
        basis_A,
        basis_B
    ):
        angles = subspace_angles(
            basis_A,
            basis_B
        )

        return np.degrees(angles)

    # --------------------------------------------------
    # Fusion novelty
    # --------------------------------------------------

    def fusion_novelty(
        self,
        ct_basis,
        pet_basis,
        fusion_basis
    ):
        """
        Measures how much fusion tangent
        lies outside CT+PET span.
        """

        union_basis = np.concatenate(
            [ct_basis, pet_basis],
            axis=1
        )

        Q, _ = np.linalg.qr(union_basis)

        projection = Q @ Q.T @ fusion_basis

        residual = (
            fusion_basis -
            projection
        )

        novelty = (
            np.linalg.norm(residual)
            /
            np.linalg.norm(fusion_basis)
        )

        return novelty

    # --------------------------------------------------
    # Full patient analysis
    # --------------------------------------------------

    def analyze_sample(self, idx):

        ct_basis = self.estimate_tangent_space(
            self.ct,
            idx
        )

        pet_basis = self.estimate_tangent_space(
            self.pet,
            idx
        )

        fusion_basis = self.estimate_tangent_space(
            self.fusion,
            idx
        )

        results = {}

        results["CT_vs_PET_Grassmann"] = (
            self.grassmann_distance(
                ct_basis,
                pet_basis
            )
        )

        results["CT_vs_Fusion_Grassmann"] = (
            self.grassmann_distance(
                ct_basis,
                fusion_basis
            )
        )

        results["PET_vs_Fusion_Grassmann"] = (
            self.grassmann_distance(
                pet_basis,
                fusion_basis
            )
        )

        results["CT_PET_Angles"] = (
            self.principal_angles(
                ct_basis,
                pet_basis
            )
        )

        results["Fusion_Novelty"] = (
            self.fusion_novelty(
                ct_basis,
                pet_basis,
                fusion_basis
            )
        )

        return results
    


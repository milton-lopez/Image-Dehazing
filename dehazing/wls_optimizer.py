import numpy as np
from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import spsolve
from skimage.color import rgb2gray


class WLSOptimizer:
    def __init__(self, lambda_=0.1, small_num=1e-5):
        self.lambda_ = lambda_
        self.small_num = small_num

    def compute_affinities(self, image):
        """Compute horizontal and vertical affinities based on image gradients."""
        dy = np.diff(image, axis=0)
        dy = -self.lambda_ / (np.abs(dy) ** 2 + self.small_num)
        dy = np.pad(dy, ((0, 1), (0, 0)), mode="constant").flatten("F")

        dx = np.diff(image, axis=1)
        dx = -self.lambda_ / (np.abs(dx) ** 2 + self.small_num)
        dx = np.pad(dx, ((0, 0), (0, 1)), mode="constant").flatten("F")

        return dx, dy

    def construct_laplacian_matrix(self, dx, dy, h, k):
        """Construct a spatially inhomogeneous Laplacian matrix."""
        diagonal_indices = np.array([-h, -1])
        tmp = spdiags(np.vstack((dx, dy)), diagonal_indices, k, k)

        east = dx
        west = np.pad(dx, (h, 0), mode="constant")[:-h]
        south = dy
        north = np.pad(dy, (1, 0), mode="constant")[:-1]
        diagonal = -(east + west + south + north)

        return tmp + tmp.T + spdiags(diagonal, 0, k, k)

    def optimize(self, input_image, data_weight, guidance):
        h, w = input_image.shape
        k = h * w

        guidance_gray = rgb2gray(guidance)
        dx, dy = self.compute_affinities(guidance_gray)
        Asmoothness = self.construct_laplacian_matrix(dx, dy, h, k)

        # Normalize data weight
        data_weight -= np.min(data_weight)
        data_weight /= np.max(data_weight) + self.small_num

        # Adjust data weight and input based on reliability
        min_in_row = np.min(input_image, axis=0)
        reliability_mask = data_weight[0, :] < 0.6
        data_weight[0, reliability_mask] = 0.8
        input_image[0, reliability_mask] = min_in_row[reliability_mask]

        Adata = spdiags(data_weight.flatten("F"), 0, k, k)
        A = Adata + Asmoothness
        b = Adata * input_image.flatten("F")

        # Solve the linear system using the CSR-formatted matrix
        out = spsolve(csr_matrix(A), b).reshape(h, w, order="F")

        return out

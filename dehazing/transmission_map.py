import numpy as np
from .wls_optimizer import WLSOptimizer


class TransmissionMap:
    def __init__(self, n_points=1000, lambda_=0.1):
        self.n_points = n_points
        self.lambda_ = lambda_
        self.wls_optimizer = WLSOptimizer(lambda_=lambda_)

    def estimate_initial_transmission(self, radius, ind):
        # Estimate radius as the maximal radius in each haze-line (Eq. (11))
        radius_max = np.bincount(
            ind.flatten(), radius.flatten(), minlength=self.n_points
        )

        # Handle cases where a haze-line has no points
        mask = radius_max == 0
        if np.any(mask):
            radius_max[mask] = np.max(radius)

        radius_new = radius_max[ind].reshape(radius.shape)

        # Estimate transmission as radii ratio (Eq. (12))
        transmission_estimation = radius / radius_new

        # Limit the transmission to the range [trans_min, 1] for numerical stability
        trans_min = 0.1
        transmission_estimation = np.clip(transmission_estimation, trans_min, 1)

        return transmission_estimation

    def compute_trans_lower_bound(self, img_hazy, air_light):
        """Compute the lower bound for transmission based on the hazy image and air light."""
        return 1 - np.min(img_hazy / np.reshape(air_light, (1, 1, 3)), axis=2)

    def compute_bin_count_map(self, ind, h, w):
        """Compute the bin count map used for reliability evaluation."""
        SMALL_BIN_THRESHOLD = 50
        bin_count = np.bincount(ind.flatten(), minlength=self.n_points)
        bin_count_map = bin_count[ind].reshape(h, w)
        return np.minimum(1, bin_count_map / SMALL_BIN_THRESHOLD)

    def compute_radius_std(self, ind, radius, h, w):
        """Compute the standard deviation of radius, used as the data-term weight."""
        STD_LOWER_BOUND = 0.001
        STD_UPPER_BOUND = 0.1
        RADIUS_SCALE_FACTOR = 3

        radius_flat = radius.flatten()
        K_std = np.zeros(self.n_points)

        for i in range(self.n_points):
            mask = ind.flatten() == i
            if np.any(mask):
                K_std[i] = np.std(radius_flat[mask])

        radius_std = K_std[ind].reshape(h, w)
        radius_std_normalized = radius_std / np.max(radius_std)

        return np.minimum(
            1,
            RADIUS_SCALE_FACTOR
            * np.maximum(STD_LOWER_BOUND, radius_std_normalized - STD_UPPER_BOUND),
        )

    def regularize_transmission(
        self, transmission_estimation, img_hazy, ind, radius, air_light
    ):
        h, w = transmission_estimation.shape

        trans_lower_bound = self.compute_trans_lower_bound(img_hazy, air_light)
        transmission_estimation = np.maximum(transmission_estimation, trans_lower_bound)

        bin_count_map = self.compute_bin_count_map(ind, h, w)
        radius_std = self.compute_radius_std(ind, radius, h, w)
        data_term_weight = bin_count_map * radius_std

        transmission = self.wls_optimizer.optimize(
            transmission_estimation, data_term_weight, img_hazy
        )

        return transmission

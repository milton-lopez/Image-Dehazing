import numpy as np
from sklearn.cluster import KMeans


class AirLightEstimator:
    def __init__(
        self,
        a_min=(0, 0.05, 0.1),
        a_max=(1, 1, 1),
        n_clusters=1000,
        spacing=0.02,
        n_angles=40,
        threshold=0.01,
    ):
        self.a_min = a_min
        self.a_max = a_max
        self.n_clusters = n_clusters
        self.spacing = spacing
        self.n_angles = n_angles
        self.threshold = threshold

    def generate_a_values(self, Avals1, Avals2):
        """
        Generate a list of air-light candidates of 2-channels, using two lists of
        values in a single channel each. This is the Python equivalent of the MATLAB
        function generate_Avals.
        """
        A1 = np.repeat(Avals1, len(Avals2))
        A2 = np.tile(Avals2, len(Avals1))
        Aall = np.column_stack((A1, A2))
        return Aall

    def vote_2d(self, points, points_weight, directions_all, candidates):
        """
        Votes for candidate points based on their alignment with given points and directions.
        Parameters:
        points: ndarray
            Array of points.
        points_weight: ndarray
            Weight of each point.
        directions_all: ndarray
            Array of direction vectors.
        candidates: ndarray
            Array of candidate points for voting.
        Returns:
        Aout: ndarray
            Candidate point with the highest weighted votes.
        accumulator_unique: ndarray
            Array of vote counts for each candidate point.
        """
        n_directions = directions_all.shape[0]
        accumulator_votes_idx = np.zeros(
            (len(candidates), len(points), n_directions), dtype=bool
        )

        # Iterate over points and directions
        for i_point, point in enumerate(points):
            for i_direction, direction in enumerate(directions_all):
                idx_to_use = np.where(
                    (candidates[:, 0] > point[0]) & (candidates[:, 1] > point[1])
                )[0]
                if not idx_to_use.size:
                    continue
                candidates_subset = candidates[idx_to_use]
                dist_factor = (
                    np.sqrt(((candidates_subset - point) ** 2).sum(axis=1)) / np.sqrt(2)
                    + 1
                )
                dist_to_line = (
                    -point[0] * direction[1]
                    + point[1] * direction[0]
                    + candidates_subset[:, 0] * direction[1]
                    - candidates_subset[:, 1] * direction[0]
                )
                idx = np.abs(dist_to_line) < 2 * self.threshold * dist_factor
                if np.any(idx):
                    accumulator_votes_idx[idx_to_use[idx], i_point, i_direction] = True

        # Ensure each vote is supported by at least 2 points
        accumulator_votes_idx = np.sum(accumulator_votes_idx, axis=2) >= 2

        # Integrate a weighted voting mechanism
        accumulator_unique = np.zeros(len(candidates))
        for i_candidate, candidate in enumerate(candidates):
            idx_to_use = np.where(
                (candidate[0] > points[:, 0]) & (candidate[1] > points[:, 1])
            )[0]
            if not idx_to_use.size:
                continue
            points_dist = np.sqrt(((candidate - points[idx_to_use]) ** 2).sum(axis=1))
            points_weight_dist = points_weight[idx_to_use] * (
                5 * np.exp(-points_dist) + 1
            )
            accumulator_unique[i_candidate] = np.sum(
                points_weight_dist[accumulator_votes_idx[i_candidate, idx_to_use]]
            )

        # Find the candidate with the highest weighted votes
        Aestimate_idx = np.argmax(accumulator_unique)
        Aout = candidates[Aestimate_idx]

        return Aout, accumulator_unique

    def cluster_colors(self, img):
        """
        Convert the input image to an indexed image using color clustering.
        Args:
            img (numpy.ndarray): Input image as a 3D array (height, width, channels).
        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: 2D array of color cluster centers.
                - numpy.ndarray: 1D array of cluster weights.
        """
        h, w, _ = img.shape
        img_flat = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_clusters).fit(img_flat)
        points = kmeans.cluster_centers_
        img_ind = kmeans.labels_.reshape(h, w)

        # Remove empty clusters and adjust indices
        unique_indices = np.unique(img_ind)
        if len(unique_indices) < self.n_clusters:
            points = points[unique_indices]
            for i, idx in enumerate(unique_indices):
                img_ind[img_ind == idx] = i

        # Count the occurrences of each index (cluster's weight)
        points_weight = np.bincount(img_ind.flatten()) / (h * w)

        return points, points_weight

    def estimate_airlight(self, img):
        """
        Estimate the airlight from an input image using color clustering and voting.
        Args:
            img (numpy.ndarray): Input image as a 3D array (height, width, channels).
        Returns:
            numpy.ndarray: Estimated airlight as a 1D array with 3 elements.
        """
        # Convert input image to an indexed image (color clustering)
        points, points_weight = self.cluster_colors(img)

        # Define arrays of candidate airlight values and angles
        angle_list = np.linspace(0, np.pi, self.n_angles, endpoint=False)
        directions_all = np.column_stack((np.sin(angle_list), np.cos(angle_list)))

        # Airlight candidates in each color channel
        a_range_r = np.arange(self.a_min[0], self.a_max[0], self.spacing)
        a_range_g = np.arange(self.a_min[1], self.a_max[1], self.spacing)
        a_range_b = np.arange(self.a_min[2], self.a_max[2], self.spacing)

        # Estimate airlight in each pair of color channels
        a_all_rg = self.generate_a_values(a_range_r, a_range_g)
        _, a_vote_rg = self.vote_2d(
            points[:, 0:2], points_weight, directions_all, a_all_rg
        )

        a_all_gb = self.generate_a_values(a_range_g, a_range_b)
        _, a_vote_gb = self.vote_2d(
            points[:, 1:3], points_weight, directions_all, a_all_gb
        )

        a_all_rb = self.generate_a_values(a_range_r, a_range_b)
        _, a_vote_rb = self.vote_2d(
            points[:, [0, 2]], points_weight, directions_all, a_all_rb
        )

        # Normalize votes
        max_val = max(np.max(a_vote_rg), np.max(a_vote_gb), np.max(a_vote_rb))
        a_vote_rg_normalized = a_vote_rg / max_val if max_val > 0 else a_vote_rg
        a_vote_gb_normalized = a_vote_gb / max_val if max_val > 0 else a_vote_gb
        a_vote_rb_normalized = a_vote_rb / max_val if max_val > 0 else a_vote_rb

        # Correctly reshape the 3D volumes
        a_vote_rg_3d = a_vote_rg_normalized.reshape(len(a_range_r), len(a_range_g), 1)
        a_vote_gb_3d = a_vote_gb_normalized.reshape(1, len(a_range_g), len(a_range_b))
        a_vote_rb_3d = a_vote_rb_normalized.reshape(len(a_range_r), 1, len(a_range_b))

        # Combine the votes
        combined_votes = a_vote_rg_3d * a_vote_gb_3d * a_vote_rb_3d

        # Find the most probable airlight
        most_probable_idx = np.unravel_index(
            np.argmax(combined_votes), combined_votes.shape
        )
        most_probable_airlight = [
            a_range_r[most_probable_idx[0]],
            a_range_g[most_probable_idx[1]],
            a_range_b[most_probable_idx[2]],
        ]

        return np.array(most_probable_airlight)

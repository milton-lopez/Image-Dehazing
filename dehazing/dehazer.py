import numpy as np
from scipy.spatial import KDTree
from .transmission_map import TransmissionMap
from .config import TESSELLATION_FILE_PATH


class Dehazer:
    def __init__(self, n_points=1000, gamma=1):
        self.n_points = n_points
        self.gamma = gamma
        self.transmission_map = TransmissionMap(n_points=n_points)

    def adjust_contrast(self, img, percentiles=[0.01, 0.99], contrast_factor=0.2):
        """
        Adjust the contrast of an image.
        Parameters:
        img: ndarray
            Input image.
        percentiles: list, optional
            Lower and upper percentile values to clip the image. Defaults to [0.01, 0.99].
        contrast_factor: float, optional
            Factor to adjust the contrast. Defaults to 0.2.
        Returns:
        ndarray
            The image with adjusted contrast.
        """
        # Validate the percentiles and contrast_factor
        if not (0 <= percentiles[0] < percentiles[1] <= 1):
            raise ValueError(
                "Percentiles should be between 0 and 1 and in ascending order."
            )
        if not (0 <= contrast_factor <= 1):
            raise ValueError("Contrast factor should be between 0 and 1.")

        # Convert percentiles to actual percentile values
        low, high = np.percentile(
            img, [percentiles[0] * 100, percentiles[1] * 100], axis=(0, 1)
        )

        # Adjust the low and high thresholds
        high_adjusted = contrast_factor * high + (1 - contrast_factor) * np.maximum(
            high, np.mean(high)
        )
        low_adjusted = contrast_factor * low + (1 - contrast_factor) * np.minimum(
            low, np.mean(low)
        )

        # Clip and normalize the image
        img_clipped = np.clip(img, low_adjusted, high_adjusted)
        img_normalized = (img_clipped - low_adjusted) / (high_adjusted - low_adjusted)

        # Scale back to the original range
        img_rescaled = img_normalized * (high_adjusted - low_adjusted) + low_adjusted

        return img_rescaled

    def find_haze_lines(self, img_hazy_corrected, air_light):
        """
        Calculate the haze lines from a corrected hazy image.
        Parameters:
        img_hazy_corrected: ndarray
            Corrected hazy image.
        air_light: ndarray
            Air light value.
        Returns:
        tuple: A tuple containing:
            - radius: The radius of each point from the air light in the image.
            - ind: The index of the closest point in the uniform tessellation.
        """
        h, w, _ = img_hazy_corrected.shape

        # Translate coordinate system to be air_light-centric
        dist_from_airlight = img_hazy_corrected - air_light

        # Calculate radius
        radius = np.linalg.norm(dist_from_airlight, axis=2)

        # Flatten and normalize the distance for clustering
        dist_unit_radius = dist_from_airlight.reshape(-1, 3)
        dist_norm = np.linalg.norm(dist_unit_radius, axis=1, keepdims=True)
        dist_unit_radius /= dist_norm

        # Load pre-calculated uniform tessellation of the unit-sphere
        tessellation_file = TESSELLATION_FILE_PATH.format(n_points=self.n_points)
        try:
            points = np.loadtxt(tessellation_file)
        except IOError:
            raise FileNotFoundError(f"Unable to find or read '{tessellation_file}'.")

        kdtree = KDTree(points)
        _, ind = kdtree.query(dist_unit_radius)

        return radius, ind

    def dehaze(self, img_hazy_corrected, transmission, air_light):
        """
        Dehaze an image using the transmission map.
        Parameters:
        img_hazy_corrected: ndarray
            Corrected hazy image.
        transmission: ndarray
            Estimated transmission of the image.
        air_light: ndarray
            Air light value.
        Returns:
        ndarray
            Dehazed image.
        """
        TRANSMISSION_MIN = 0.1
        LEAVE_HAZE_FACTOR = (
            1.06  # Leave a bit of haze for a natural look (set to 1 to reduce all haze)
        )
        ADJUST_PERCENT = [0.005, 0.995]

        air_light_reshaped = air_light.reshape(1, 1, 3)
        img_dehazed = (
            img_hazy_corrected
            - (1 - LEAVE_HAZE_FACTOR * transmission[..., np.newaxis])
            * air_light_reshaped
        ) / np.maximum(transmission[..., np.newaxis], TRANSMISSION_MIN)

        img_dehazed = np.clip(img_dehazed, 0, 1)
        img_dehazed = np.power(img_dehazed, 1 / self.gamma)  # radiometric correction
        img_dehazed = self.adjust_contrast(img_dehazed, ADJUST_PERCENT)

        return (img_dehazed * 255).astype(np.uint8)

    def non_local_dehazing(self, img_hazy, air_light):
        """
        Perform non-local dehazing on an RGB image.
        Parameters:
        img_hazy: ndarray
            Input hazy RGB image.
        air_light: ndarray
            Air light value in RGB.
        Returns:
        tuple: Dehazed image and the transmission.
        """
        if img_hazy.shape[-1] != 3:
            raise ValueError(
                f"Expected a 3-channel (RGB) image, but received {img_hazy.shape[-1]} channels."
            )
        if air_light is None or len(air_light) != 3:
            raise ValueError(
                "air_light must be a 3-element array representing RGB airlight."
            )

        img_hazy = img_hazy.astype(np.float64) / 255.0
        img_hazy_corrected = np.power(img_hazy, self.gamma)  # radiometric correction

        radius, ind = self.find_haze_lines(img_hazy_corrected, air_light)
        transmission_estimation = self.transmission_map.estimate_initial_transmission(
            radius, ind
        )
        transmission = self.transmission_map.regularize_transmission(
            transmission_estimation, img_hazy, ind, radius, air_light
        )

        img_dehazed = self.dehaze(img_hazy_corrected, transmission, air_light)

        return img_dehazed, transmission

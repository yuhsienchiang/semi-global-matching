import numpy as np
import time
from tqdm import tqdm


class MatchCost:
    def __init__(self, img_L, img_R):
        self.img_L = np.array(img_L, dtype=np.float64)
        self.img_R = np.array(img_R, dtype=np.float64)

    def compute(self, disparity_level=5, window_size=3):
        """Compute the matching cost usinng normalised cross correlation method

        Args:
            disparity_level (int, optional):
                The maximum disparity to search. Defaults to 5.
            window_size (int, optional):
                The width of the square matching window. The value should be an odd integer. Defaults to 3.

        Returns:
            disparity_map (float, numpy.ndarray, shape(H, W)):
                A matrix of disparity values of each pixel in the image
            cost_map (float, numpy.ndarray, shape(H, W, disparity_level+1)):
                The cost value for each position (height, width, disparity)
        """
        # start timing
        start = time.time()

        # Initialise parameters
        H, W = self.img_L.shape
        disparity_level += 1
        # the radius of the window
        half_window_size = window_size // 2
        cost_map = np.zeros((H, W, disparity_level), dtype=np.float64)

        for h in tqdm(range(half_window_size, H - half_window_size, 1)):

            for w in range(half_window_size + disparity_level, W - half_window_size, 1):

                for d in range(disparity_level):
                    if w - half_window_size - d >= 0:
                        cost_map[h, w, d] = -self.normalise_cross_correlate(p_L=(h, w), p_R=(h, w - d), window_size=window_size)
                    else:
                        cost_map[h, w, d] = -self.normalise_cross_correlate(p_L=(h, w), p_R=(h, 0), window_size=window_size)

        self.cost_map = cost_map
        self.disparity_map = np.argmin(cost_map, axis=2)

        # Record function runtime
        end = time.time()
        self.runtime = end - start

        return self.disparity_map, self.cost_map

    def normalise_cross_correlate(self, p_L, p_R, window_size):
        """Calculate the normalised cross correlation of the two windows
           in img_L and img_R at the position of p_L and p_R

        Args:
            p_L ((int, int)):
                The cooridinate of the window center at img_L.
                The first element is the height, the second element is the widths
            p_R ((int, int)):
                the cooridinate of the window center at img_R.
                The first element is the height, the second element is the widths
            window_size (int, odd number):
                The dimension of the square window.  The value should be an odd integer

        Returns:
            normalised_cross_correlation (float):
                The normalised cross correlation of the two windows
                in img_L and img_R at the position of p_L and p_R.  The value is a float >= 0
        """
        h_L, w_L = p_L
        h_R, w_R = p_R

        half_window_size = window_size // 2

        matrix_L = self.img_L[
            h_L - half_window_size : h_L + half_window_size + 1,
            w_L - half_window_size : w_L + half_window_size + 1,
        ]
        matrix_R = self.img_R[
            h_R - half_window_size : h_R + half_window_size + 1,
            w_R - half_window_size : w_R + half_window_size + 1,
        ]

        # Shift the matrix so that the mean of the matrix is 0
        matrix_L = matrix_L - np.mean(matrix_L, dtype=np.float64)
        matrix_R = matrix_R - np.mean(matrix_R, dtype=np.float64)

        # Compute the scaling factor of the matrix
        norm_L = np.linalg.norm(matrix_L)
        norm_R = np.linalg.norm(matrix_R)

        # return normalised cross correlation value, if the norm of either matrix is 0 and two matrices are not identical, return 0
        # if the norm of either matrix is 0 but the two matrices are identical, return 1
        cc = np.sum((matrix_L * matrix_R), dtype=np.float64)
        if norm_L != 0 and norm_R != 0:
            cc = np.sum((matrix_L * matrix_R))
            ncc = cc / (norm_L * norm_R)
        elif np.array_equal(matrix_L, matrix_R):
            ncc = 1
        else:
            ncc = 0

        return ncc


class MatchCost_Cache:
    def __init__(self, img_L, img_R):
        self.img_L = np.array(img_L, dtype=np.float64)
        self.img_R = np.array(img_R, dtype=np.float64)

    def compute(self, disparity_level=5, window_size=3):
        """Compute the matching cost usinng normalised cross correlation method

        Args:
            disparity_level (int, optional):
                The maximum disparity to search. Defaults to 5.
            window_size (int, optional):
                The width of the square matching window. The value should be an odd integer. Defaults to 3.

        Returns:
            disparity_map (float, numpy.ndarray, shape(H, W)):
                A matrix of disparity values of each pixel in the image
            cost_map (float, numpy.ndarray, shape(H, W, disparity_level+1)):
                The cost value for each position (height, width, disparity)
        """

        # start timing
        start = time.time()

        # Set up parameters
        H, W = self.img_L.shape
        disparity_level += 1
        half_window_size = window_size // 2

        # Initialise cost map
        cost_map = np.zeros((H, W, disparity_level), dtype=np.float64)

        # Caches
        img_L_mean_cache, img_L_scalar_cache = self.matrix_cache(self.img_L, window_size)

        img_R_shift = np.copy(self.img_R)
        img_R_mean_cache, img_R_scalar_cache = self.matrix_cache(img_R_shift, window_size)

        # Loop through disparity
        for d in tqdm(range(disparity_level)):

            # cross correlation cache
            cross_corr_cache = self.img_L * img_R_shift

            for h in range(half_window_size, H - half_window_size):

                for w in range(half_window_size + d, W - half_window_size):

                    window_L = self.img_L[
                        h - half_window_size : h + half_window_size + 1,
                        w - half_window_size : w + half_window_size + 1,
                    ]
                    window_R = img_R_shift[
                        h - half_window_size : h + half_window_size + 1,
                        w - half_window_size : w + half_window_size + 1,
                    ]

                    cross_corr = cross_corr_cache[
                        h - half_window_size : h + half_window_size + 1,
                        w - half_window_size : w + half_window_size + 1,
                    ]

                    l = window_L * img_R_mean_cache[h, w]
                    r = window_R * img_L_mean_cache[h, w]

                    norm_l = img_L_scalar_cache[h, w]
                    norm_r = img_R_scalar_cache[h, w]

                    if norm_l != 0 and norm_r != 0:
                        cost_map[h, w, d] = -np.sum(cross_corr- l - r + (img_L_mean_cache[h, w] * img_R_mean_cache[h, w])) / (norm_l * norm_r)
                    elif np.array_equal(window_L - img_L_mean_cache[h, w], window_R - img_R_mean_cache[h, w]):
                        cost_map[h, w, d] = -1
                    else:
                        cost_map[h, w, d] = 0

            # shift R image caches to the right for calculating cost value for next disparity
            img_R_shift = np.roll(img_R_shift, 1, axis=1)
            img_R_shift[:, 0] = 0
            img_R_mean_cache = np.roll(img_R_mean_cache, 1, axis=1)
            img_R_mean_cache[:, 0] = 0
            img_R_scalar_cache = np.roll(img_R_scalar_cache, 1, axis=1)
            img_R_scalar_cache[:, 0] = 0

        self.cost_map = cost_map
        self.disparity_map = np.argmin(cost_map, axis=2)

        # Recoard function runtime
        end = time.time()
        self.runtime = end - start

        return self.disparity_map, self.cost_map

    def matrix_cache(self, img, window_size):
        H, W = img.shape
        half_window_size = window_size // 2

        img_mean_cache = np.zeros((H, W), dtype=np.float64)
        img_scalar_cache = np.zeros((H, W), dtype=np.float64)

        for h in range(half_window_size, H - half_window_size):
            for w in range(half_window_size, W - half_window_size):
                window_matrix = img[
                    h - half_window_size : h + half_window_size + 1,
                    w - half_window_size : w + half_window_size + 1,
                ]
                mean = np.mean(window_matrix, dtype=np.float64)

                img_mean_cache[h, w] = mean
                img_scalar_cache[h, w] = np.linalg.norm(window_matrix - mean)

        return img_mean_cache, img_scalar_cache
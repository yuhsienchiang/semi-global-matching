import time
import numpy as np


class SGM:

    def __init__(self, p1, p2, cost_map=None):
        self.cost_map = np.array(cost_map, np.float64)
        self.p1 = p1
        self.p2 = p2

    def compute(self, cost_map=None):
        """Implementing semi-global matching algorithm to compute the smoothed cost value for a given cost volum matrix

        Args:
            cost_map (float, numpy.ndarray, shape(H, W, disparity_level)):
                The cost map can be the result of any matching algorithm . Defaults to None.

        Returns:
            disparity_map (float, numpy.ndarray, shape(H, W)):
                A matrix of disparity values of each pixel in the image
            energy_matrix (float, numpy.ndarray, shape(H, W, disparity_level+1)):
                The cost value for each position (height, width, disparity)
        """

        # start timinng
        start = time.time()

        if cost_map is not None:
            self.cost_map = np.array(cost_map, dtype=np.float64)

        # Initialise parameters
        H, W, D = self.cost_map.shape
        E_init = np.zeros((H, W, D), dtype=np.float64)
        E_init[0, :, :] = self.cost_map[0, :, :]
        E_init[H - 1, :, :] = self.cost_map[H - 1, :, :]
        E_init[:, 0, :] = self.cost_map[:, 0, :]
        E_init[:, W - 1, :] = self.cost_map[:, W - 1, :]

        # Left to Right scan
        # L2R
        E_a1 = np.copy(E_init)
        E_a1[:, 1:W] = 0

        # TL2BR
        E_a5 = np.copy(E_init)
        E_a5[1:H, 1:W] = 0

        # BL2TR
        E_a8 = np.copy(E_init)
        E_a8[0 : H - 1, 1:W] = 0

        for w in range(1, W):
            E_a1[:, w, :] = self.cost_map[:, w, :] + self.cost_arggregate(E_pre=E_a1[:, w - 1, :])

            c_m = self.cost_map[:, w, :]
            c_m[1:H, :] = c_m[1:H, :] + self.cost_arggregate(E_pre=E_a5[:, w - 1, :])[0 : H - 1, :]
            E_a5[:, w, :] = c_m

            c_m = self.cost_map[:, w, :]
            c_m[0 : H - 1, :] = c_m[0 : H - 1, :] + self.cost_arggregate(E_pre=E_a8[:, w - 1, :])[1:H, :]
            E_a8[:, w, :] = c_m

        # Right to Left scan
        # R2L
        E_a2 = np.copy(E_init)
        E_a2[:, 0 : W - 1] = 0

        # TR2BL
        E_a6 = np.copy(E_init)
        E_a6[1:H, 0 : W - 1] = 0

        # BR2TL
        E_a7 = np.copy(E_init)
        E_a7[0 : H - 1, 0 : W - 1] = 0

        for w in range(W - 2, -1, -1):
            E_a2[:, w, :] = self.cost_map[:, w, :] + self.cost_arggregate(E_pre=E_a2[:, w + 1, :])

            c_m = self.cost_map[:, w, :]
            c_m[1:H, :] = c_m[1:H, :] + self.cost_arggregate(E_pre=E_a6[:, w + 1, :])[0 : H - 1, :]
            E_a6[:, w, :] = c_m

            c_m = self.cost_map[:, w, :]
            c_m[0 : H - 1, :] = c_m[0 : H - 1, :] + self.cost_arggregate(E_pre=E_a7[:, w + 1, :])[1:H, :]
            E_a7[:, w, :] = c_m

        # Top to Bottom scan
        # T2B
        E_a3 = np.copy(E_init)
        E_a3[1:H, :] = 0
        for h in range(1, H):
            E_a3[h, :, :] = self.cost_map[h, :, :] + self.cost_arggregate(E_pre=E_a3[h - 1, :, :])

        # Bottom to Top scan
        # B2T
        E_a4 = np.copy(E_init)
        E_a4[0 : H - 1, :] = 0
        for h in range(H - 2, -1, -1):
            E_a4[h, :, :] = self.cost_map[h, :, :] + self.cost_arggregate(E_pre=E_a4[h + 1, :, :])

        # Aggregate teh costs
        E = np.array(E_a1 + E_a2 + E_a3 + E_a4 + (E_a5 + E_a6 + E_a7 + E_a8) * 2.0, dtype=np.int64)

        self.energy_matrix = E
        self.disparity_map = np.argmin(E, axis=2)

        # Recoard function runtime
        end = time.time()
        self.runtime = end - start

        return self.disparity_map, self.energy_matrix

    def cost_arggregate(self, E_pre):
        """Computing the aggregated cost value of a specific H or W position

        Args:
            E_pre (float, numpy.ndarray, shape(R, disparity_level+1)):
                The cost values of the previous position

        Returns:
            aggregate_cost_map (float, numpy.ndarray, shape(R, disparity_level+1)):
                The aggregated cost value map
        """

        R, D = E_pre.shape

        # Shifting the array along the disarpity axis in positive direction
        d_plus = np.full((R, D), np.inf, dtype=np.float64)
        d_plus[:, 1:D] = E_pre[:, 0 : D - 1] + self.p1

        # Shifting the array along the disarpity axis in negative direction
        d_minus = np.full((R, D), np.inf, dtype=np.float64)
        d_minus[:, 0 : D - 1] = E_pre[:, 1:D] + self.p1

        # Find the minimum cost value along the disparity axis
        d_min = np.repeat([np.amin(E_pre, axis=1)], D, axis=0).T + self.p2

        return np.amin([E_pre, d_plus, d_minus, d_min], axis=0) - (d_min - self.p2)
import numpy as np
from tqdm import tqdm

class MatchCost():
    def __init__(self, img_L, img_R):
        self.img_L = img_L
        self.img_R = img_R
    
    def compute(self, disparity_level=5, window_size=3):
        half_window_size = window_size // 2
        H, W = self.img_L.shape
        
        cost_map = np.zeros((H, W, disparity_level), dtype=np.float64)
        
        for h in tqdm(range(half_window_size, H-half_window_size, 1)):
            
            for w in range(half_window_size+ disparity_level, W-half_window_size, 1):
                
                for d in range(disparity_level):
                    if w-half_window_size-d >= 0:
                        cost_map[h,w,d] = -self.normalise_cross_correlate(p_L=(h, w), p_R=(h, w - d), window_size=window_size)
                    else:
                        cost_map[h,w,d] = -self.normalise_cross_correlate(p_L=(h, w), p_R=(h, w), window_size=window_size)
        
        self.cost_map = cost_map
        self.disparity_map = np.argmin(cost_map, axis=2)
        
        return self.disparity_map, self.cost_map
    
    
    def normalise_cross_correlate(self, p_L, p_R, window_size):
        h_L, w_L = p_L
        h_R, w_R = p_R
        
        half_window_size = window_size // 2
        
        matrix_L = self.img_L[h_L - half_window_size : h_L + half_window_size + 1, w_L - half_window_size : w_L + half_window_size + 1]
        matrix_R = self.img_R[h_R - half_window_size : h_R + half_window_size + 1, w_R - half_window_size : w_R + half_window_size + 1]
        
        matrix_L = matrix_L - np.mean(matrix_L)
        matrix_R = matrix_R - np.mean(matrix_R)

        norm_L = np.linalg.norm(matrix_L)
        norm_R = np.linalg.norm(matrix_R)
        return np.sum((matrix_L * matrix_R))/(norm_L * norm_R) if norm_L and norm_R !=0 else 0
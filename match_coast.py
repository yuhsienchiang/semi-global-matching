import numpy as np
import cv2


class MatchCost():
    def __init__(self, img_L, img_R):
        self.img_L = img_L
        self.img_R = img_R
        self.window_size = None
        self.disparity_level = None
        
    
    def compute(self, disparity_level=5, window_size=3):
        self.disparity_level = disparity_level
        self.window_size = window_size
        half_window_size = window_size // 2
        
        h, w = self.img_L.shape
        
        disparity_map = np.zeros((h, w, disparity_level), dtype=np.float64)
        
        for y in range(start=half_window_size, stop=h-half_window_size, step=1):
            
            for x in range(start=half_window_size ,stop=w-half_window_size, step=1):
                
                for d in range(disparity_level):
                    disparity_map[y,x,d] = -self.normalise_cross_correlate(p_L=(x, y), p_R=(x-d, y), window_size=self.window_size)
        
        return disparity_map
    
    
    def normalise_cross_correlate(self, p_L, p_R, window_size):
        x_L, y_L = p_L
        x_R, y_R = p_R
        
        half_window_size = window_size // 2
        
        matrix_L = self.img_L[y_L - half_window_size : y_L + half_window_size + 1 , x_L - half_window_size : x_L + half_window_size + 1]
        matrix_R = self.img_R[y_R - half_window_size : y_R + half_window_size + 1 , x_R - half_window_size : x_R + half_window_size + 1]
        
        nor_matrix_L = self.normalise(matrix_L)
        nor_matrix_R = self.normalise(matrix_R)
        
        return nor_matrix_L * nor_matrix_R
        
        
    def normalise(self, array):
       
       mean = np.mean(array, dtype=np.float64)
       std = np.linalg.norm(mean - std)
       
       return ((array-mean)/std)
        
        
                
        
    


import numpy as np
import math
from scipy.io import loadmat, savemat
from array import array

class BFM:
	def __init__(self):
		model_path = './BFM/BFM_model_front.mat'
        model = loadmat(model_path)
        
        self.mean_shape = model['meanshape'].T.astype(np.float32) # mean face shape 
        self.id_base = model['idBase'].astype(np.float32) # identity basis
        self.ex_base = model['exBase'].astype(np.float32) # expression basis
        self.mean_tex = model['meantex'].T.astype(np.float32) # mean face texture
        self.tex_base = model['texBase'].astype(np.float32) # texture basis
        
        self.point_buf = model['point_buf'].astype(np.int64) - 1 # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
        self.tri = model['tri'].astype(np.int64) - 1 # vertex index for each triangle face, starts from 1
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1 # 68 face landmark index, starts from 0
        
        self.a0 = np.pi
        self.a1 = 2*np.pi/np.sqrt(3.)
        self.a2 = 2*np.pi/math.sqrt(8.0)
        self.c0 = 1/math.sqrt(4*math.pi)
        self.c1 = math.sqrt(3.0)/math.sqrt(4*math.pi)
        self.c2 = 3*math.sqrt(5.0)/math.sqrt(12*math.pi)


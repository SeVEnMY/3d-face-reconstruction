import numpy as np
from scipy.io import loadmat, savemat
from array import array


def perspective_projection(focal, center):
  # return p.T (N, 3) @ (3, 3) 
  return np.array([
      focal, 0, center,
      0, focal, center,
      0, 0, 1
  ]).reshape([3, 3]).astype(np.float32).transpose()

class BFM:
	def __init__(self):
		model_path = './BFM/BFM_model_front.mat'
    model = loadmat(model_path)
    
    self.mean_shape = model['meanshape'].astype(np.float32) # mean face shape 
    self.id_base = model['idBase'].astype(np.float32) # identity basis
    self.ex_base = model['exBase'].astype(np.float32) # expression basis
    self.mean_tex = model['meantex'].astype(np.float32) # mean face texture
    self.tex_base = model['texBase'].astype(np.float32) # texture basis
    
    self.point_buf = model['point_buf'].astype(np.int64) - 1 # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
    self.tri = model['tri'].astype(np.int64) - 1 # vertex index for each triangle face, starts from 1
    self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1 # 68 face landmark index, starts from 0

    self.persc_proj = perspective_projection(1015., 112.)
    self.device='cpu'
    self.camera_distance=10
    self.init_lit = np.array(
        [0.8, 0, 0, 0, 0, 0, 0, 0, 0]
        ).reshape([1, 1, -1]).astype(np.float32)

    self.a0 = np.pi
    self.a1 = 2*np.pi/np.sqrt(3.)
    self.a2 = 2*np.pi/np.sqrt(8.0)
    self.c0 = 1/np.sqrt(4*np.pi)
    self.c1 = np.sqrt(3.0)/np.sqrt(4*np.pi)
    self.c2 = 3*np.sqrt(5.0)/np.sqrt(12*np.pi)

    def to(self, device):
      self.device = device
      for key, value in self.__dict__.items():
        if type(value).__module__ == np.__name__:
          setattr(self, key, torch.tensor(value).to(device))

    def compute_shape(self, id_coeff, exp_coeff):
      batch_size = id_coeff.shape[0]
      id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
      exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
      face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
      return face_shape.reshape([batch_size, -1, 3])

    def compute_texture(self, tex_coeff, normalize=True):
      batch_size = tex_coeff.shape[0]
      face_texture = torch.einsum('ij,aj->ai', self.tex_base, tex_coeff) + self.mean_tex
      if normalize:
          face_texture = face_texture / 255.
      return face_texture.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
      v1 = face_shape[:, self.face_buf[:, 0]]
      v2 = face_shape[:, self.face_buf[:, 1]]
      v3 = face_shape[:, self.face_buf[:, 2]]
      e1 = v1 - v2
      e2 = v2 - v3
      face_norm = torch.cross(e1, e2, dim=-1)
      face_norm = F.normalize(face_norm, dim=-1, p=2)
      face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)
      
      vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
      vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
      return vertex_norm

  def compute_light(self, face_texture, face_norm, gamma):
      batch_size = gamma.shape[0]
      v_num = face_texture.shape[1]
      gamma = gamma.reshape([batch_size, 3, 9])
      gamma = gamma + self.init_lit
      gamma = gamma.permute(0, 2, 1)
      Y = torch.cat([
           a0 * c0 * torch.ones_like(face_norm[..., :1]).to(self.device),
          -a1 * c1 * face_norm[..., 1:2],
           a1 * c1 * face_norm[..., 2:],
          -a1 * c1 * face_norm[..., :1],
           a2 * c2 * face_norm[..., :1] * face_norm[..., 1:2],
          -a2 * c2 * face_norm[..., 1:2] * face_norm[..., 2:],
          0.5 * a2 * c2 / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
          -a2 * c2 * face_norm[..., :1] * face_norm[..., 2:],
          0.5 * a2 * c2 * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
      ], dim=-1)
      r = Y @ gamma[..., :1]
      g = Y @ gamma[..., 1:2]
      b = Y @ gamma[..., 2:]
      face_color = torch.cat([r, g, b], dim=-1) * face_texture
      return face_color

    def to_camera(self, face_shape):
      face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
      return face_shape

    def to_image(self, face_shape):
      face_proj = face_shape @ self.persc_proj
      face_proj = face_proj[..., :2] / face_proj[..., 2:]

      return face_proj

    def get_landmarks(self, face_proj):
      return face_proj[:, self.keypoints]

    def split_coeff(self, coeffs):
      id_coeffs = coeffs[:, :80]
      exp_coeffs = coeffs[:, 80: 144]
      tex_coeffs = coeffs[:, 144: 224]
      gammas = coeffs[:, 224: 251]
      return {
          'id': id_coeffs,
          'exp': exp_coeffs,
          'tex': tex_coeffs,
          'gamma': gammas,
      }

    def compute_for_render(self, coeffs):
      coef_dict = self.split_coeff(coeffs)
      face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])

      face_vertex = self.to_camera(face_shape)
      
      face_proj = self.to_image(face_vertex)
      landmark = self.get_landmarks(face_proj)

      face_texture = self.compute_texture(coef_dict['tex'])
      face_norm = self.compute_norm(face_shape)
      face_light = self.compute_light(face_texture, face_norm, coef_dict['gamma'])

      return face_vertex, face_texture, face_light, landmark
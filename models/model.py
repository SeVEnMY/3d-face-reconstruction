import numpy as np
import torchvision
import torch
from torch import nn
import soft_renderer as sr
from .losses import img_loss, landmark_loss
from .bfm import BFM
import trimesh

class ReconModel:
	def __init__(self, is_train):
    self.isTrain = is_train
    self.device = torch.device('gpu') 
    self.save_dir = os.path.join('./checkpoints', '111')  # save all the checkpoints to save_dir
    self.loss_names = []
    self.image_paths = []
		self.model_names=['face_recon']
		self.visual_names=['visualization']
		self.parallel_names = self.model_names + ['renderer']
    self.resnet50 = torchvision.models.resnet50(pretrained=True)
    self.resnet50.fc = nn.Linear(resnet50.fc.in_features, 251)

		self.net_recon = self.resnet50

		self.face_model = BFM()
		self.renderer = sr.SoftRenderer(image_size=224, sigma_val=1e-4, aggr_func_rgb='hard', 
                            camera_mode='look_at', viewing_angle=30, fill_back=False,
                            perspective=False, light_intensity_ambient=1.0, light_intensity_directionals=0)

		if self.isTrain:
			self.net_recog = self.resnet50
			self.loss_names = ['all', 'lm', 'img']
			self.comupte_img_loss = img_loss
			self.compute_lm_loss = landmark_loss

        self.optimizer = torch.optim.Adam(self.net_recon.parameters(), lr=1e-3)
        self.optimizers = [self.optimizer]
        self.parallel_names += ['net_recog']

  def set_input(self, input):
  	self.input_img = input['imgs'].to(self.device) 
    self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
    self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
    self.trans_m = input['M'].to(self.device) if 'M' in input else None
    self.image_paths = input['im_paths'] if 'im_paths' in input else None

  def forward(self):
  	coeffs = self.net_recon(self.input_img)
  	self.face_model.to(self.device)
  	self.pred_vertex, self.pred_tex, self.pred_light, self.pred_lm = \
  		self.face_model.compute_for_render(coeffs)
  	self.pred_coeffs_dict = self.face_model.split_coeff(output_coeff)

  	self.renderer = sr.SoftRenderer(image_size=224, sigma_val=1e-4, aggr_func_rgb='hard',
                            camera_mode='look_at', viewing_angle=30, fill_back=False,
                            perspective=False, light_intensity_ambient=1.0, light_intensity_directionals=0)
  	self.pred_img = self.renderer(torch.Tensor(self.face_vertex[None,:,:,]).cuda(),
               torch.Tensor(np.flip(self.face_model.tri-1, axis=1).copy()).cuda(),
               torch.Tensor(self.pred_tex[None,:,:]).cuda(),
               texture_type="vertex")[0].permute(1,2,0)

  def compute_losses(self):
  	assert self.net_recog.training == False
  	self.loss_img = 0.5*self.comupte_img_loss(self.pred_img, self.input_img, self.atten_mask)
  	self.loss_lm = 0.5*self.compute_lm_loss(self.pred_lm, self.gt_lm)
  	self.loss_all = loss_img+loss_lm

  def optimize_parameters(self, isTrain=True):
        self.forward()               
        self.compute_losses()
        """Update network weights; it will be called in every training iteration."""
        if isTrain:
            self.optimizer.zero_grad()  
            self.loss_all.backward()         
            self.optimizer.step()        

    def compute_visuals(self):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            
            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
            
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy, 
                                    output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(
                    output_vis_numpy / 255., dtype=torch.float32
                ).permute(0, 3, 1, 2).to(self.device)

    def save_mesh(self, name):

        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        mesh.export(name)

    def save_coeff(self,name):

        pred_coeffs = {key:self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:,:,0],self.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        savemat(name,pred_coeffs)
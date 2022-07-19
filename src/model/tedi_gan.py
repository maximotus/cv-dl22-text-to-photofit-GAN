from error import ConfigurationError
from torch.optim import Adam, Adagrad, SGD
from torch.nn import BCELoss, CrossEntropyLoss
import torch
from torch import nn
from model.helper import Tedi_Generator, GradualStyleBlock, BackboneEncoderUsingLastLayerIntoW, BackboneEncoderUsingLastLayerIntoWPlus
import logging

logger = logging.getLogger('root')
VALID_OPTIMIZER_NAMES = {'Adam': Adam, 'Adagrad': Adagrad, 'SGD': SGD}
VALID_CRITERION_NAMES = {'BCELoss': BCELoss, 'CrossEntropyLoss': CrossEntropyLoss}

# class TediGAN:
#     def __init__(self, model_params, optimizer_name, image_size, learning_rate, device_name):

        

#         # TODO initialize optimizer
#         # self.optimizer = VALID_OPTIMIZER_NAMES[optimizer_name](self.parameters(), lr=learning_rate)

#         # TODO
#         raise NotImplementedError


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class TediGAN(nn.Module):

	def __init__(self, model_params, optimizer_name, learning_rate, criterion_name, num_classes, device):
		super(TediGAN, self).__init__()
		if optimizer_name not in VALID_OPTIMIZER_NAMES:
			raise ConfigurationError(
                'Specified optimizer is not valid. Valid optimizers: ' + str(VALID_OPTIMIZER_NAMES))
		
		logger.info('Initializing TediGAN...')

		# initialize parameters
		self.dropout = model_params.get('dropout')
		self.alpha = model_params.get('alpha')
		self.beta1 = model_params.get('beta1')
		self.ngf = model_params.get('ngf')
		self.ndf = model_params.get('ndf')
		self.use_spectral_norm = model_params.get('use_spectral_norm')
		self.z_channels = model_params.get('z_channels')
		self.num_classes = num_classes
		self.lr = learning_rate
		self.device = device

		# initialize loss function
		if criterion_name not in VALID_CRITERION_NAMES:
			raise ConfigurationError(
				'Specified criterion is not valid. Valid loss functions: ' + str(VALID_CRITERION_NAMES))
		self.criterion = VALID_CRITERION_NAMES[criterion_name]()

		# Define architecture
		self.encoder = GradualStyleBlock(50, 'ir_se')#self.set_encoder()
		# self.decoder = Tedi_Generator(self.ngf, self.ndf, self.emb_dim, lr_mlp=self.lr)#1024, 512, 8
		self.decoder = Tedi_Generator(1024, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()
		

	# def set_encoder(self):
	# 	if self.opts.encoder_type == 'GradualStyleEncoder':
	# 		encoder = GradualStyleBlock(50, 'ir_se', self.opts)
	# 	elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
	# 		encoder = BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
	# 	elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
	# 		encoder = BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
	# 	else:
	# 		raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
	# 	return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load('./stylegan2-ffhq-config-f.pt')
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=18)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
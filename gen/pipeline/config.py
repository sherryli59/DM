from yacs.config import CfgNode as CN

cfg = CN()

cfg.device="cuda:0"

cfg.dataset = CN()
cfg.dataset.name = None
cfg.dataset.potential = None
cfg.dataset.centers = None
cfg.dataset.std = None
cfg.dataset.training_data = None
cfg.dataset.testing_data = None
cfg.dataset.data = None
cfg.dataset.size = 32
cfg.dataset.nchannels = 2
cfg.dataset.dim = 2
cfg.dataset.nparticles = None
cfg.dataset.shape = None
cfg.dataset.boxlength = None
cfg.dataset.sigma = 1
cfg.dataset.epsilon = 1
cfg.dataset.kT = 1.0
cfg.dataset.cutoff = None
cfg.dataset.shift = True
cfg.dataset.periodic = True
cfg.dataset.atomic_numbers = None
cfg.dataset.dependency = None
cfg.dataset.translation_inv = False

cfg.model = CN()
cfg.model.type = "diffusion"  # diffusion or nf
cfg.model.representation = CN() 
cfg.model.representation.type = None
cfg.model.representation.num_blocks = 1
cfg.model.representation.hidden_dim = 1000
cfg.model.representation.context_dim = 0
# for diffusion model
cfg.model.sde = CN()
cfg.model.sde.type = None
cfg.model.sde.knots = []
cfg.model.sde.schedule = "linear"
cfg.model.sde.friction = 1.0

# for nf model
cfg.model.flow = CN()
cfg.model.flow.type = "continuous" # continuous or autoregressive
cfg.model.prior = CN()
cfg.model.prior.type = "Normal"
cfg.model.prior.shape = None
cfg.model.prior.std = None
cfg.model.prior.cutoff = None
cfg.model.prior.centers = None

cfg.train_parameters = CN()
cfg.train_parameters.max_epochs = 4000
cfg.train_parameters.batch_size = 100
cfg.train_parameters.output_freq = 100
cfg.train_parameters.scheduler = "exponential"
cfg.train_parameters.lr_scheduler_gamma = 0.999

cfg.optimizer = CN()
cfg.optimizer.lr = 1e-4
cfg.optimizer.beta1 = 0.9
cfg.optimizer.eps = 1e-8
cfg.optimizer.warmup = 0 
cfg.optimizer.grad_clip = 1.


cfg.output=CN()
cfg.output.training_dir="../training/"
cfg.output.testing_dir="../testing/"
cfg.output.model_dir="../saved_models/"
cfg.output.best_model_dir = "../trained_models/"


def get_cfg_defaults():
  return cfg.clone()

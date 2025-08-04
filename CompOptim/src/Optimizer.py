from .CommNet import CommNet
from .Compressor import *

'''
The Optimizer class implements various different methods for optimization.
'''
class Optimizer(CommNet):

	def __init__(self,model,compressor,optim_name="DistributedAdam",comm_set=['x'],device="cpu",topology="ring",devices=[],nvlink=False,lr_decay="none",lr=0.001, model_name = ""):
		super().__init__(topology=topology,comms=device,devices=devices,nvlink=nvlink,compressor=NoneCompressor())
		
		self.comm_set = comm_set
		self.optim_name = optim_name
		self.model= model
		self.opt_compressor = compressor
		
		# for cosine decay as in NanoGPT
		self.lr=lr
		self.lr_decay = lr_decay
		self.steps = 0
		self.warmup_iters = 2000 #100 for all but hb and cd, 2000
		self.min_lr = 6e-5 #1e-4,6e-5
		self.lr_decay_iters = 5000 #2000, 5000
		
		self.model_name = model_name

		if device == "cpu":
			self.device = device
		else:
			self.device = devices[self.rank % len(devices)]


	def get_names(self,field):
		return self.data[field].keys()

	def get_lr(self, it):
		# 1) linear warmup for warmup_iters steps
		# if it%500 == 0 and it<=2000:
		if it < self.warmup_iters:
			return self.lr * it / self.warmup_iters
		# 2) if it > lr_decay_iters, return min learning rate
		if it > self.lr_decay_iters:
			return self.min_lr
		# 3) in between, use cosine decay down to min learning rate
		decay_ratio = (it - self.warmup_iters) / \
				(self.lr_decay_iters - self.warmup_iters)
		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
		return self.min_lr + coeff * (self.lr - self.min_lr)

					
	'''
    Performs a conditional neighbor_reduce() from CommNet for the given variable.
    If the given variable is not in the "comm_set" list, then nothing happens. 
    NOTE:
    	- This only works on one subfield / "name" at a time. This allows for it to be done
    	  within each loop of "step()".
    '''
	def neighbor_reduce_cond(self,field,name,comm_set=[]):
		if self.nprocs == 1:
			return
		
		# If we are communicating the current field, perform a neighbor_reduce and set the associated values.
		if field in comm_set and self.epoch:
			super().neighbor_reduce(field,name,unique=False)

			self.data[field][name] = self.recv_data["reduced"][name].clone()
	
	def state_dict(self):
		state = {
			'optim_name': self.optim_name,
			'lr':          self.lr,
			'steps':       self.steps,
			'eps':         self.eps,
			'epoch':       self.epoch,
			'beta1':       self.beta1,
			'data':        {
				field: {name: tensor.detach().cpu()
						for name, tensor in tensors.items()}
				for field, tensors in self.data.items()
			}
		}
		return state

	def load_state_dict(self, state):
		for key, value in state.items():
			if key == 'data':
				continue
			setattr(self, key, value)

		for field in state['data']:
			if field not in self.data:
				self.data[field] = {}
			for name, tensor in state['data'][field].items():
				self.data[field][name] = tensor.to(self.device)

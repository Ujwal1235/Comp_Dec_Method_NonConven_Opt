import os
import torch
from torch.utils.data import DataLoader as DL
from torch.utils.data.sampler import SubsetRandomSampler as SRS
import models
import torch.nn as nn

from .read_datasets import read_datasets
from .Tracker import Tracker
from .system_info import nvlink_check
from .system_info import get_cuda_devices
from .Compressor import *

from optimizers import get_optimizer
'''
DistDataModel is a class which deals with simulating data distributed models.
The class inherets from CommNet, which defines a network topology and communication
methods to be used. This child class deals with the following components.
- Defining a model on each MPI rank.
- Defines the train and test loaders.
- Seeds the models.
- Defines distributed training methods.
- Contains evaluators for results.

NOTES:
	- Need to enable setting the seed. Currently the parameter does nothing.
'''
class DistDataModel():

	def __init__(self,model="LeNet5",dataset="FashionMNIST",topology="ring",\
		optimizer=None,comm_set=['x'],batch_size=16,device='cpu',\
		nvlink=False,track=True,seed=1337,compressor=NoneCompressor(), variety="index", lr_decay="none", lr = 0.001, resume=False):

		# Define our training duration and the communication parameters.
		self.epoch = 0
		self.epochs = 50
		self.comm_set = comm_set
		self.topology = topology
		self.seed = seed

		# Define our model, and optimizers.
		self.device = device
		self.optimizer_name = optimizer
		self.model_name = model
		self.model = models.__dict__[model]()
		self.lr_decay = lr_decay
		self.lr = lr
		self.resume = resume 		#if previous checkpoint present to resume training from
		self.set_optimizer(optimizer,compressor,nvlink=nvlink)
		self.model = self.model.to(self.device)
		self.dataset = dataset
		self.variety = variety

		self.nprocs = self.optim.nprocs
		self.rank = self.optim.rank
		self.train_dataset, self.test_dataset = read_datasets(dataset)
		self.batch_size = batch_size
		self.loss_fcn = nn.CrossEntropyLoss()
		self.form_loaders(variety=self.variety)
		self.best_val_loss = 1e9

		# Set up our tracker if it makes sense.
		self.track = track
		if track == True:
			self.tracker = Tracker(model=self.model,model_name=self.model_name,loss_function=self.loss_fcn,\
			test_loader=self.test_loader,train_loader=self.train_loader,device=self.device)

	'''
	Gets the samples for each rank from the data set and defines our data loaders.
	'''
	def form_loaders(self,variety="index"):

		if variety == "index":

			# First we have to define our set of indices to split the data and get our sampler.
			data_per_node = int(len(self.train_dataset)/self.nprocs)
			idx = [data_per_node*self.rank+i for i in range(data_per_node)]

		elif variety == "label":

			class_num = len(self.train_dataset.classes)
			class_name = [self.rank,class_num-self.rank-1]
			idx =  []
			for i in range(len(self.train_dataset)):
				if self.train_dataset[i][1] in class_name:
					idx.append(i)

		else:
			print("ERROR IN FORM LOADER (RANK: {}). INVALID VARIETY ({}).".format(self.rank,variety))

		# Set our Sampler.
		sampler = SRS(idx)

		# Now we define our loaders based on this.
		self.train_loader = DL(self.train_dataset, batch_size=self.batch_size, sampler=sampler, drop_last=True)
		self.test_loader = DL(self.test_dataset, batch_size=1000, shuffle=False)

	'''
	Sets our optimizer based on an input string.
	'''
	def set_optimizer(self,optimizer_name,compressor,nvlink=False):

		if self.device == "cpu":
			device_list=[]
			self.optim = get_optimizer(optimizer_name,self.model, compressor, \
							  comm_set=self.comm_set, device=self.device, topology=self.topology, \
							  devices=device_list, nvlink=nvlink, lr_decay=self.lr_decay,lr=self.lr)
			
		else:
			# Get the names of our cuda enabled devices.
			devices = get_cuda_devices()

			# If our nvlink flag is live, check if our system has an nvlink using nvidia-smi.
			if nvlink:
				nvlink = nvlink_check()

			self.optim = get_optimizer(optimizer_name,self.model, compressor, \
							  comm_set=self.comm_set, device=self.device, topology=self.topology, \
							  devices=devices, nvlink=nvlink, lr_decay=self.lr_decay,lr=self.lr)
			
			if self.resume == True:
				ckpt_path = "" # input the checkpoint path to continue the training from
				checkpoint = torch.load(ckpt_path)
				self.optim.load_state_dict(checkpoint['optimizer'])
				self.epoch = checkpoint['iter_num']
				self.best_val_loss = checkpoint['best_val_loss']
			rank = self.optim.rank
			self.device = self.optim.devices[rank % len(self.optim.devices)]

	'''
	Performs training for the defined number of epochs.
	'''
	def train(self,output_file="default",verbose=False):

		if self.rank == 0:
			if self.model_name=="nanoGPT":
				print("epoch\tRank\ttest_loss\ttrain_loss\tcons_error\ttest_time\ttrain_time")
			else:
				print("epoch\tRank\ttest_acc\ttest_loss\ttrain_acc\ttrain_loss\tcons_error\ttest_time\ttrain_time")

		if self.track:
			verbose = True

		while self.epoch < self.epochs:
			# Increment our epoch and set the model to train.
			self.epoch += 1
			self.model.train()

			# If verbose, print our stats.
			if self.track:
				test_loss,test_acc,test_time = self.tracker.evaluate(loader="test")
				cons_error = self.tracker.compute_cons_error(self.comm_set,self.optim)
				train_loss,train_acc,train_time = self.tracker.evaluate(loader="train")
				if verbose:
					if self.model_name=="nanoGPT":
						print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.epoch-1,self.rank,test_loss,train_loss,cons_error,test_time,train_time))
					else:
						print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.epoch-1,self.rank,test_acc,test_loss,train_acc,train_loss,cons_error,test_time,train_time))
			loss = None

			# Perform training for our batches.
			for _, (data,target) in enumerate(self.train_loader):
				# Do the standard training stuff.
				data,target = data.to(self.device), target.to(self.device)
				self.model.zero_grad()
				if self.model_name=="nanoGPT":
					output,loss = self.model(data,target)
				else:
					output = self.model(data)
					loss = self.loss_fcn(output, target)
				loss.backward()
				
				# Perform step.
				self.optim.step()

			# Set up a barrier at the end of the epoch.
			self.optim.COMM.Barrier()
		
			losses = torch.zeros(self.epochs)
			for k in range(self.epochs):
				losses[k] = loss.item()
			losses_mean = losses.mean()
			if losses_mean < self.best_val_loss:
				self.best_val_loss = losses_mean
				if self.epoch % 25==0:
					checkpoint = {
						'model': self.model.state_dict(),
						'optimizer': self.optim.state_dict(),
						'iter_num': self.epoch,
						'best_val_loss': self.best_val_loss
					}
					if self.model_name == "nanoGPT":
						checkpoint.update({'model_args':self.model.model_args, 'config': self.model.config})
					os.makedirs(self.model.out_dir, exist_ok=True)
					torch.save(checkpoint, os.path.join(self.model.out_dir, 'chk_'+output_file+'_Epoch_'+str(self.epoch)+'.pt'))
			print(f"Checkpoints saved at {self.model.out_dir}")

		# If we are tracking, return our out_dict at the end of training.
		if self.track:
			self.tracker.save_history(self.dataset+output_file,self.optim.COMM)
			return(self.tracker.history)
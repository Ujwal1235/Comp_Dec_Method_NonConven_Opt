import torch
import yaml
import numpy as np
import time
from .CommNet import CommNet

class Tracker(CommNet):

	def __init__(self,model,loss_function,test_loader,train_loader,device,model_type):
		self.model = model
		self.loss_fcn = loss_function
		self.device = device
		self.test_loader = test_loader
		self.train_loader = train_loader
		self.model_type = model_type
		keys = ["test_loss","test_acc","train_loss","train_acc","train_time","test_time","cons_error"]
		self.history = {key:[] for key in keys}

	'''
	Gets the loss and accuracy for reporting.
	'''
	def evaluate(self,loader="test",output=True):

		# Set the correct data loader.
		data_loader = 0
		if loader == "test":
			data_loader = self.test_loader
		elif loader == "train":
			data_loader = self.train_loader
		else:
			print("INVALID DATA LOADER IN :: DistDataModel.evaluate()")
			return( -1, -1 )

		# First we define a dictionary for tracking values.
		stats = {"loss":[],"acc":[]}

		# Start the timer.
		start_time = time.time()

		# Loop over the test loader and track our evaluation metrics.
		for _, (data, target) in enumerate(data_loader):
			data, target = data.to(self.device), target.to(self.device)
			self.model.zero_grad()
			if self.model_type == "nanoGPT":
				_ ,loss = self.model(data,target)
				#because we cant really compute an accuracy, we will set it to some value i.e. -1
				acc  = -1
				if loss == None:
					loss = torch.tensor(0, dtype=torch.int8)
				stats['loss'].append(loss.data.detach().cpu())
				stats['acc'].append(acc.data.detach().cpu())

			else:
				model_output = self.model(data)
				loss = self.loss_fcn(model_output, target)
				pred = model_output.argmax(dim=1, keepdim=True)
				acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
				stats['loss'].append(loss.data.detach().cpu())
				stats['acc'].append(acc.data.detach().cpu())

		# End the timer and calculate the elapsed time.
		end_time = time.time()
		elapsed_time = end_time - start_time

		# Save the information to self.history. If output is true, return the values.
		if loader == "test":
			self.history["test_loss"].append(float(np.mean(np.array(stats['loss']))))
			self.history["test_acc"].append(float(np.mean(np.array(stats['acc']))))
			self.history["test_time"].append(elapsed_time)
			if output:
				return(self.history["test_loss"][-1],self.history["test_acc"][-1],self.history["test_time"][-1])
			
		elif loader == "train":
			self.history["train_loss"].append(float(np.mean(np.array(stats['loss']))))
			self.history["train_acc"].append(float(np.mean(np.array(stats['acc']))))
			self.history["train_time"].append(elapsed_time)
			if output:
				return(self.history["train_loss"][-1],self.history["train_acc"][-1],self.history["train_time"][-1])
			
		else:
			print("INVALID DATA LOADER IN :: DistDataModel.evaluate()")
			return( -1, -1 )

    #helper to compute the cons error
	def compute_cons_error(self,comm_set,optimizer):
		c_err = []
		err=0
		for field in comm_set:
			for name in optimizer.get_names(field):
				ws_mean, w = optimizer.neighbor_reduce(field,name)
				err = pow(torch.norm(w-ws_mean,'fro'),2).cpu()
				c_err.append(err)

		#perform true allRE

		self.history["cons_error"].append(float(np.mean(np.array(c_err))))


		return np.mean(np.array(c_err))
		#return cons

	'''
	Saves the data to a single file.
	NOTES:
		- This is a jank way to work around the fact that our established CommNet is behind our
		optimizer. However it only needs to be done once, at the end of training, so it's fiiiine.
	'''
	def save_history(self,file_name,COMM_WORLD_INSTANCE):

		# Set up our file path.
		path = "./results/"+file_name+".yaml"

		# Hijack the COMM_WORLD to perform an all_gather.
		gathered_history = COMM_WORLD_INSTANCE.allgather(self.history)

		# Check our current rank. If it is 0, we dump to a yaml.
		if COMM_WORLD_INSTANCE.Get_rank() == 0:
			file = open(path,"w")
			yaml_string = yaml.dump(gathered_history,file)
			file.close()

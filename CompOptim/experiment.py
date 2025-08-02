from src.DistDataModel import DistDataModel
import os
import yaml
from yaml import Loader
from src.Compressor import *
from src.DaSHCo import *

compressor_map = {'none': NoneCompressor(),
				   'topk30': TopKCompressor(0.3),
				   'topk40': TopKCompressor(0.4),
				   'topk50': TopKCompressor(0.5),
				   'topk60': TopKCompressor(0.6),
				   'qsgd': QSGDCompressor(2)}


def convert_to_compress(comm_set):
	BAR_STRING = "_bar"
	return list(x+BAR_STRING for x in comm_set)

# First we construct our data distributed neural network.
#GPU optimizations

RANK = 4
dataset = ["FashionMNIST"]#["FashionMNIST","CIFAR10","Shakespeare"]
compress_method =["topk30"]
variety_type = ["index"]
lr_decay= ["none"]
optimizers = [("NewAlg",['x'],0.001)]#("NewAlg",['x'],0.001),("DistributedAdaGrad",['x'],0.001),("DistributedAdam",['x'],0.001)]
topology = ["ring"]
k_list = [1]

mod = "LeNet5" #"LeNet5"
bs=8 #8, 64

models=[]
names = []
epchs = 150


for data in dataset:
	for compress in compress_method:
		for opt,comm_set,lr in optimizers:
			for lr_type in lr_decay:
				for variety in variety_type:
					for k in k_list:
					#when we compress using the new algorithms, we want to communicate these new terms
						if "topk" in compress:
							comm_set = ['x_bar']#convert_to_compress(comm_set)

						models.append(DistDataModel(model=mod,dataset=data,topology="ring",optimizer=opt,\
						comm_set=comm_set,batch_size=bs,device="cpu",track=True,seed=1337,\
						compressor=compressor_map[compress],lr_decay=lr_type,variety=variety,learning_rate=lr,k=k))

						names.append(data+"-"+opt+"-"+compress+"-Rank-"+str(RANK)+"-"+variety+"-lrtype-"+lr_type+"-K-"+str(k)+"-"+str(epchs))


if __name__ == "__main__":
	for i in range(len(models)):
		e_model = models[i]
		e_model.epochs=epchs*e_model.k
		print(names[i])
		print("Model initialized.... Now Starting training....",flush=True)
		training_history = e_model.train(verbose=True,output_file=names[i])
		print("training finished...",flush=True)

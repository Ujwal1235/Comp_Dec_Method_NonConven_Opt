from mpi4py import MPI
import torch
from .Compressor import *
# from .temp_compressor import *

'''
CommNet is a class which deals with multiple aspects of distributed models.
The class contains methods for:
- Defining a communication network.
- Sending vectors throughout the network.
- Communication scheduling.
'''

'''
NOTES:
- Need a way to easily define the topology with a type check for string.
- setup_neighbors can ONLY do looping definitions at the moment. Cannot do standard lattice.
	To add to this. We could also add a dictionary of tuples, but this is probably too much.
'''
class CommNet:

	def __init__(self,topology="ring",comms="cpu",devices=[],nvlink=False,compressor=NoneCompressor()):
		self.topology = topology
		self.data = {}
		self.recv_data = {}
		self.devices = devices
		self.nvlink = nvlink
		self.compressor = compressor

		if comms == "cpu" or comms == "gpu":
			self.comms = comms
		else:
			print("WARNING: Invalid 'comms' argument in CommNet initialization ({})".format(comms))
		self.setup_neighbors()

	'''
	Performs the original setup of the network neighbors and 
	determines the number of nodes in the network.
	'''
	def setup_neighbors(self):

		# Set up the COMM_WORLD for cpu communication (Always performed)
		self.COMM = MPI.COMM_WORLD
		self.rank = self.COMM.Get_rank()
		self.nprocs = self.COMM.Get_size()

		# Based on self.topology and self.nprocs, get the neighbors.
		self.neighbors = []
		if self.topology == "ring":
			values = [-1,1]
			for value in values:
				self.neighbors.append((self.rank + value) % self.nprocs)
				self.recv_data[(self.rank + value) % self.nprocs] = {}

		# If the number of ranks is different than the number of GPU, and we are not
		# doing CPU-only comms, then we throw a warning.
		if self.comms == "gpu":
			if self.nprocs > (len(self.devices)):
				print("WARNING: In CommNet.setup_neighbors() :: self.nprocs ({}) != GPU-count ({}).".format(self.nprocs,len(self.devices)))

	'''
	Basic send and receive calls.
	'''
	def net_send(self, neighbor_id, field, name, tag=0, verbose=False):
		tensor = self.data[field][name]
		if self.comms == "gpu" and not self.nvlink:
			tensor = tensor.clone().to("cpu")
		comp_info = self.compressor.compress(tensor)
		payload = comp_info[1:]
		self.COMM.send(payload, dest=neighbor_id, tag=tag)
		if verbose:
			print(f"RANK: {self.rank} SENT-TO: {neighbor_id}")

	def net_recv(self, neighbor_id, name, tag=0, unique=False, verbose=False):
		payload = self.COMM.recv(source=neighbor_id, tag=tag)
		out = self.compressor.decompress(payload)
		if self.comms == "gpu" and not self.nvlink:
			gpu_id = self.devices[self.rank % len(self.devices)]
			out = out.to(gpu_id)
		if unique:
			self.recv_data[neighbor_id][name] = out
		else:
			self.recv_data["reduced"][name]  += out
		if verbose:
			print(f"RANK: {self.rank} RECEIVED-FROM: {neighbor_id}")

	'''
	Considering the topology, performs a "local all-gather" that 
	only works over neighbors. (I know this is nonsense wording, but you get it.)
	NOTES: 
		- Currently only works with a ring network and an even self.nprocs value.
		- Assumes the data being sent and received are pytorch 
			tensors of the same size.
	'''
	def neighbor_gather(self,field,name,unique=False):

		# Now we consider the topology of the network to decide
		# our communication scheme.
		if self.topology == "ring":
			n = self.nprocs

			for phase in (0, 1):
				for i in range(n):
					if i % 2 != phase:
						continue
					j = (i + 1) % n
					if self.rank == i:
						# send your data to j, then receive j’s
						self.net_send(neighbor_id=j, field=field, name=name)
						self.net_recv(neighbor_id=j, name=name, unique=unique)
					elif self.rank == j:
						# receive from i, then send back
						self.net_recv(neighbor_id=i, name=name, unique=unique)
						self.net_send(neighbor_id=i, field=field, name=name)

		self.COMM.Barrier()

	'''
	Performs a neighbor gather and then reduces the results to a reduced state.
	NOTES:
		- Currently also only works with a basic choice of mixing matrix where every node
		  is worth the same amount. (TO DO)
	'''
	def neighbor_reduce(self,field,name,unique=False):
		self.recv_data["reduced"][name] = torch.zeros_like(self.data[field][name])
		self.neighbor_gather(field, name, unique=unique)
		reduced = self.data[field][name].clone()

		if unique:
			for nbr_id, nbr_dict in self.recv_data.items():
				if nbr_id == "reduced":
					continue
				reduced.add_(nbr_dict[name])
		else:
			reduced.add_(self.recv_data["reduced"][name])

		reduced.div_(float(len(self.neighbors) + 1))

		self.recv_data["reduced"][name] = reduced
		self.COMM.Barrier()

		return reduced, self.data[field][name]

	'''
	Does an all reduce over our network.
	'''
	def all_reduce(self,field,name):

		# Perform an allgather.
		collected_data = self.COMM.allgather(self.data[field][name])
		self.recv_data["reduced"][name] = torch.zeros_like(self.data[field][name])
		for node in range(self.nprocs):
			self.recv_data["reduced"][name] += collected_data[node]/self.nprocs

		self.COMM.Barrier()

		return self.recv_data["reduced"][name],self.data[field][name]
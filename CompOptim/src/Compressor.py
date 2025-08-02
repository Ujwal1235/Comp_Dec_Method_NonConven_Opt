
######################################################################
##### Compressor, NoneCompressor, TopKCompressor
##### The code in this file is original from grace_dl/dist/compressor (__version__ = '1.0')
##### https://github.com/sands-lab/grace
######################################################################

import torch
import math
from abc import ABC, abstractmethod

class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")


#Quantization Compressor from grace_dl, modified from out exp
class QSGDCompressor(Compressor):

    def __init__(self, quantum_num):
        super().__init__()
        #this is s
        self.quantum_num = quantum_num
        self.tau = 0


    #Eta function from QSG paper

    def get_name(self):
        return "qsgd"
 
#Eta function from QSG paper
    def compress(self, tensor):
        shape = tensor.size()
        tensor = tensor.flatten()
        sign = tensor.sign()


        self.tau = 1 + min(len(tensor)/self.quantum_num**2,math.sqrt(len(tensor))/self.quantum_num)


        norm = tensor.norm()
        abs_gradient = tensor.abs()


        level_float = (self.quantum_num / norm)*abs_gradient

        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()

        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)


        tensor_compressed = (new_level * sign).type(torch.int32)

      #  tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)

        #print(tensor_compressed)
        
        return 3,(1/self.tau)*tensor_compressed,norm,shape

    def decompress(self, tensors):
        tensors, norm,shape = tensors[0],tensors[1],tensors[2]

        decode_output = tensors.type(torch.float32)
        tensor_decompressed = (norm * decode_output)/self.quantum_num
       
        tensor_decompressed = tensor_decompressed.view(shape)

 
        return tensor_decompressed
    

## None-Compressor
class NoneCompressor(Compressor):
    """Default no-op compression."""
    def compress(self, tensor):
        #return [tensor], None
        return 1,tensor

    def get_name(self):
        return "none"

    def decompress(self, tensors):
        tensor = tensors[0]
        #return tensor
        return tensor


## TopK-Compressor
class TopKCompressor(Compressor):

    def __init__(self,compress_ratio):
        self.compress_ratio = compress_ratio


    def get_name(self):
        return "topk"
   

    def compress(self, tensor):

        #flatten the tensor
        t=tensor.flatten()
        k = max(1,tensor.numel()*self.compress_ratio)
        _,idxs= torch.topk(t.abs(), int(k))


        
        return 4,t[idxs],idxs,tensor.numel(),tensor.size()

    def decompress(self, tensors):

        #tensors should be a list
        values,indices,numel,shape=tensors[0],tensors[1],tensors[2],tensors[3]
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)




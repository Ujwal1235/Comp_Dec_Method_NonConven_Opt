from src.Optimizer import Optimizer
from src.Compressor import *
import torch

class DaSHCo(Optimizer):

    def __init__(self,model,lr=0.001,compressor=NoneCompressor(),device="cpu",devices=[],comm_set=['x'],lr_decay="none",k=1,nvlink=False):
        super().__init__(model,compressor,optim_name="DaSHCo",comm_set=comm_set,device=device,topology="ring",devices=devices,
                         nvlink=nvlink,lr_decay=lr_decay,lr=lr,k=k)
        self.set_data()

    def set_data(self):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.mu = 1e-4
        self.epoch = 0
        self.gamma_g = 1
        self.gamma_x = 1

        use_compressor = (self.opt_compressor.get_name() != "none")

        self.data = {}
        self.recv_data = {"reduced": {}}

        for name, param in self.model.named_parameters():
            x = param.data.detach().clone().to(self.device).half()
            self.data["x"] = self.data.get("x", {}); self.data["x"][name] = x

            zero = torch.zeros_like(x)

            self.data["g"] = self.data.get("g", {}); self.data["g"][name] = zero.clone()
            self.data["m"] = self.data.get("m", {}); self.data["m"][name] = zero.clone()
            self.data["g_tilde"] = self.data.get("g_tilde", {}); self.data["g_tilde"][name] = zero.clone()
            self.data["g_tilde_prev"] = self.data.get("g_tilde_prev", {});  self.data["g_tilde_prev"][name] = zero.clone()

            if use_compressor:
                self.data["g_bar"] = self.data.get("g_bar", {}); self.data["g_bar"][name] = zero.clone()
                self.data["g_bar_prev"] = self.data.get("g_bar_prev", {}); self.data["g_bar_prev"][name] = zero.clone()
                self.data["delta_g"] = self.data.get("delta_g", {}); self.data["delta_g"][name] = zero.clone()
                self.data["x_bar"] = self.data.get("x_bar", {}); self.data["x_bar"][name] = zero.clone()
                self.data["x_bar_prev"] = self.data.get("x_bar_prev", {}); self.data["x_bar_prev"][name] = zero.clone()
                self.data["delta_x"] = self.data.get("delta_x", {}); self.data["delta_x"][name] = zero.clone()

            else:
                self.data["g_bar"] = self.data.get("g_bar", {}); self.data["g_bar"][name] = None
                self.data["g_bar_prev"] = self.data.get("g_bar_prev", {}); self.data["g_bar_prev"][name] = None
                self.data["delta_g"] = self.data.get("delta_g", {}); self.data["delta_g"][name] = None
                self.data["x_bar"] = self.data.get("x_bar", {}); self.data["x_bar"][name] = None
                self.data["x_bar_prev"] = self.data.get("x_bar_prev", {}); self.data["x_bar_prev"][name] = None
                self.data["delta_x"] = self.data.get("delta_g", {}); self.data["delta_g"][name] = None

            self.recv_data["reduced"] = self.recv_data.get("reduced", {}); self.recv_data["reduced"][name] = zero.clone()

    def step(self):
        self.epoch += 1
        for name,param in self.model.named_parameters():
            with torch.no_grad():
                learning_rate = self.lr
                if self.lr_decay == "cosine":
                    learning_rate = super.get_lr(self.steps)
                
                self.data['g_tilde'][name] = param.grad.data.detach().clone()
                self.data['g'][name] = self.data['g'][name] - self.data['g_tilde_prev'][name] + self.data['g_tilde'][name]
                self.data['g_tilde_prev'][name] = self.data['g_tilde'][name].clone()

                if self.opt_compressor.get_name() !="none":
                    compress_in = self.opt_compressor.compress(self.data['g'][name]-self.data['g_bar'][name])
                    self.data['g_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
                    self.data['g_bar_prev'][name] = self.data['g_bar'][name].clone()
                    compress_final = self.opt_compressor.compress(self.data['g'][name]-self.data['g_bar'][name])
                    self.data['delta_g'][name] = self.opt_compressor.decompress(compress_final[1:])
                    super.neighbor_reduce_cond(field='g_bar',name=name,comm_set=self.comm_set)
                    self.data['g'][name] = self.data['g'][name]+self.gamma_g*(self.data['g_bar'][name]-self.data['g_bar_prev'][name])

                else:
                    super.neighbor_reduce_cond(field='g',name=name,comm_set=self.comm_set)
                
                self.data['m'][name] = (self.beta1 * self.data['m'][name]) + ((1-self.beta1) * self.data['g'][name])
                self.data['x'][name] = (self.data['x'][name] - learning_rate*(self.data['m'][name]))

                
                if self.opt_compressor.get_name() != "none":
                    compress_in = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])
                    self.data['x_bar'][name] += self.opt_compressor.decompress(compress_in[1:])
                    self.data['x_bar_prev'][name] = self.data['x_bar'][name].clone()
                    compress_final = self.opt_compressor.compress(self.data['x'][name]-self.data['x_bar'][name])
                    self.data['delta_x'][name] = self.opt_compressor.decompress(compress_final[1:])
                    super.neighbor_reduce_cond(field='x_bar',name=name,comm_set=self.comm_set) 
                    self.data['x'][name] = (self.data['x'][name]+self.gamma_x*(self.data['x_bar'][name]-self.data['x_bar_prev'][name]))
            
                else:
                    super.neighbor_reduce_cond(field='x',name=name,comm_set=self.comm_set)

                param.copy_(self.data['x'][name])
                param.grad.copy_(self.data['g'][name])
                self.steps += 1

    def state_dict(self):
        state = super().state_dict()
        state.update({
            'beta2':   self.beta2,
            'gamma_g': self.gamma_g,
            'gamma_x': self.gamma_x,
            'mu':      self.mu
        })
        return state
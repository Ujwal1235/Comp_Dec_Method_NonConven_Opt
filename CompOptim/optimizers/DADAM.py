from src.Optimizer import Optimizer
from src.Compressor import *
import torch

class DADAM(Optimizer):

    def __init__(self,model,lr=0.001,compressor=NoneCompressor(),device="cpu",devices=[],comm_set=['x'],lr_decay="none",k=1,nvlink=False):
        super().__init__(model,compressor,optim_name="DistributedAdam",comm_set=comm_set,device=device,topology="ring",devices=devices,
                         nvlink=nvlink,lr_decay=lr_decay,lr=lr,k=k)
        self.set_data()

    def set_data(self):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.epoch = 0
        self.gamma = 0.5

        use_compressor = (self.opt_compressor.get_name() != "none")

        self.data = {}
        self.recv_data = {"reduced": {}}

        for name, param in self.model.named_parameters():
            x = param.data.detach().clone().to(self.device).half()
            self.data["x"] = self.data.get("x", {});       self.data["x"][name] = x

            zero = torch.zeros_like(x)

            self.data["g"] = self.data.get("g", {}); self.data["g"][name] = zero.clone()
            self.data["m"] = self.data.get("m", {}); self.data["m"][name] = zero.clone()
            self.data["v"] = self.data.get("v", {}); self.data["v"][name] = zero.clone()
            self.data["v_prev"] = self.data.get("v_prev", {}); self.data["v_prev"][name] = zero.clone()
            self.data["u"] = self.data.get("u", {}); self.data["u"][name] = zero.clone()
            self.data["u_prev"] = self.data.get("u_prev", {}); self.data["u_prev"][name] = zero.clone()

            if use_compressor:
                self.data["g_bar"] = self.data.get("g_bar", {}); self.data["g_bar"][name] = zero.clone()
                self.data["g_bar_prev"] = self.data.get("g_bar_prev", {}); self.data["g_bar_prev"][name] = zero.clone()
                self.data["x_bar"] = self.data.get("x_bar", {}); self.data["x_bar"][name] = zero.clone()
                self.data["x_bar_prev"] = self.data.get("x_bar_prev", {}); self.data["x_bar_prev"][name] = zero.clone()

            else:
                self.data["g_bar"] = self.data.get("g_bar", {}); self.data["g_bar"][name] = None
                self.data["g_bar_prev"] = self.data.get("g_bar_prev", {}); self.data["g_bar_prev"][name] = None
                self.data["x_bar"] = self.data.get("x_bar", {}); self.data["x_bar"][name] = None
                self.data["x_bar_prev"] = self.data.get("x_bar_prev", {}); self.data["x_bar_prev"][name] = None

            self.recv_data["reduced"] = self.recv_data.get("reduced", {}); self.recv_data["reduced"][name] = zero.clone()

    def step(self):
        for name, param in self.model.named_parameters():
            with torch.no_grad():
                g = param.grad.data

                if self.opt_compressor.get_name() != "none":
                    delta = g - self.data["g_bar"][name]
                    comp = self.opt_compressor.compress(delta)[1:]
                    self.data["g_bar"][name].add_(self.opt_compressor.decompress(comp))
                    buf_prev = self.data["g_bar_prev"][name]
                    buf_curr = self.data["g_bar"][name]
                    scratch  = buf_prev.new_empty(buf_prev.shape)
                    scratch.copy_(buf_prev)
                    buf_prev.copy_(buf_curr)
                    buf_curr.copy_(scratch)

                    super().neighbor_reduce_cond(field="g_bar",name=name,comm_set=self.comm_set)

                    g = g + self.gamma * (self.data["g_bar"][name]- self.data["g_bar_prev"][name])
                else:
                    super().neighbor_reduce_cond(field="g",name=name,comm_set=self.comm_set)

                self.data["g"][name] = g

                m = self.data["m"][name]
                m.mul_(self.beta1).add_(g, alpha=(1.0 - self.beta1))
                super().neighbor_reduce_cond(field="m",name=name,comm_set=self.comm_set)

                v = self.data["v"][name]
                v_prev = self.data["v_prev"][name]
                v_prev.copy_(v)
                v.mul_(self.beta2).addcmul_(g, g, value=(1.0 - self.beta2))
                super().neighbor_reduce_cond(field="v",name=name,comm_set=self.comm_set)

                u = self.data["u"][name]
                u_prev = self.data["u_prev"][name]
                u_prev.copy_(u)
                u.sub_(v_prev).add_(v)
                super().neighbor_reduce_cond(field="u",name=name,comm_set=self.comm_set)

                m.div_(u_prev.add(self.eps).sqrt_())

                lr_val = (super().get_lr(self.steps)
                        if self.lr_decay == "cosine"
                        else self.lr)
                x = self.data["x"][name]
                x.add_(m, alpha=-lr_val)

                if self.opt_compressor.get_name() != "none":
                    delta_x = x - self.data["x_bar"][name]
                    comp_x = self.opt_compressor.compress(delta_x)[1:]
                    self.data["x_bar"][name].add_(self.opt_compressor.decompress(comp_x))
                    buf_prev = self.data["x_bar_prev"][name]
                    buf_curr = self.data["x_bar"][name]
                    scratch  = buf_prev.new_empty(buf_prev.shape)
                    scratch.copy_(buf_prev)
                    buf_prev.copy_(buf_curr)
                    buf_curr.copy_(scratch)

                    super().neighbor_reduce_cond(field="x_bar",name=name,comm_set=self.comm_set)

                    x.add_(self.data["x_bar"][name]- self.data["x_bar_prev"][name],alpha=self.gamma)
                else:
                    super().neighbor_reduce_cond(field="x",name=name,comm_set=self.comm_set)

                self.data["x"][name] = x
                param.copy_(x)
                param.grad.copy_(g)

                self.steps += 1

    def state_dict(self):
        state = super().state_dict()
        state['beta2'] = self.beta2
        state['gamma'] = self.gamma
        return state
import utilities

import base64
import io
import time
import pickle
import math
import numpy as np
import pylab as pl
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

import packaging.version
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
  raise RuntimeError('Torch versions lower than 1.5.0 not supported')


######################## Real NVP flow


def make_checker_mask(shape, parity,torch_device):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker.to(torch_device)



class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity, torch_device):
        super().__init__()
        self.mask = make_checker_mask(mask_shape, mask_parity, torch_device)
        self.net = net

    def forward(self, x):
        x_frozen = self.mask * x      
        x_active = (1 - self.mask) * x
        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))
        return fx, logJ

    def reverse(self, fx):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx  
        net_out = self.net(fx_frozen.unsqueeze(1))
        s, t = net_out[:,0], net_out[:,1]
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask)*(-s), dim=tuple(axes))
        return x, logJ
    
    
    
def make_conv_net(*, hidden_sizes, kernel_size, in_channels, out_channels, use_final_tanh, padding_mode):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    net = []
    for i in range(len(sizes) - 1):
        padding_size = (kernel_size[i]//2)
        net.append(torch.nn.Conv2d(
            sizes[i], sizes[i+1], kernel_size[i], padding=padding_size,
            stride=1, padding_mode=padding_mode))
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)



"""class Bias(torch.nn.Module):
    def __init__(self, torch_device):
        super().__init__()
        self.b0 = nn.Parameter(torch.randn(1) / 5.).to(torch_device)

    def forward(self, x):
        fx = x - self.b0
        return fx, 0

    def reverse(self, fx):
        x = fx + self.b0
        return x, 0"""
    


def make_flow1_affine_layers(*, n_layers, lattice_shape, hidden_sizes, kernel_size, torch_device, padding_mode, bias=False):
    layers = []
    #bias_layers = []
    
    for i in range(n_layers):
        parity = i % 2
        net = make_conv_net(
            in_channels=1, out_channels=2, hidden_sizes=hidden_sizes,
            kernel_size=kernel_size, use_final_tanh=True, padding_mode=padding_mode)
        coupling = AffineCoupling(
            net, mask_shape=lattice_shape, mask_parity=parity, torch_device=torch_device)
        
        #if bias:
        #    bias_layers.append(Bias(torch_device)) #bias at beginning, later at end
        #    layers.append(bias_layers[i])
            
        layers.append(coupling)
    
    #if bias:
    #    bias_layers.append(Bias(torch_device))
    #    layers.append(bias_layers[n_layers]) #one bias at end; now layers are symmetric
        
    return torch.nn.ModuleList(layers)


######################## Priors


class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)
    
    
    
class CorrelatedNormal:
    def __init__(self, loc, var,nx, dx,cl_theo,torch_device):
        self.torch_device=torch_device
        self.nx=nx
        self.dx=dx
        
        #normal distribution to draw random fourier modes
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.rfourier_shape = loc.shape
        
        #create the array to multiply the fft with to get the desired power spectrum
        self.ells_flat = self.get_ell(self.nx, self.dx).flatten() 
        clfactor = np.interp( self.ells_flat, np.arange(0,len(cl_theo)), np.sqrt(cl_theo), right=0 ).reshape( self.rfourier_shape[0:2] )
        self.clfactor = torch.from_numpy(clfactor).float().to(torch_device)
        clinvfactor = np.copy(clfactor) 
        clinvfactor[clinvfactor==0] = 1. #TODO: should we to remove the monopole?
        clinvfactor = 1./clinvfactor
        self.clinvfactor = torch.from_numpy(clinvfactor).float().to(torch_device)
        
        #masks for rfft symmetries
        a_mask = np.ones((self.nx, int(self.nx/2+1)), dtype=bool)
        a_mask[int(self.nx/2+1):, 0] = False
        a_mask[int(self.nx/2+1):, int(nx/2)] = False
        b_mask = np.ones((self.nx, int(self.nx/2+1)), dtype=bool)    
        b_mask[0,0] = False
        b_mask[0,int(self.nx/2)] = False
        b_mask[int(self.nx/2),0] = False
        b_mask[int(self.nx/2),int(self.nx/2)] = False
        b_mask[int(self.nx/2+1):, 0] = False
        b_mask[int(self.nx/2+1):, int(self.nx/2)] = False
        self.a_mask = a_mask
        self.b_mask = b_mask
        
        #how many mask elements
        a_nr = self.a_mask.sum()
        b_nr = self.b_mask.sum()
        #print (a_nr,b_nr)

        #make distributions with the right number of elements for each re and im mode.
        a_shape = (a_nr)
        loc = torch.zeros(a_shape)
        var = torch.ones(a_shape)
        self.a_dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        b_shape = (b_nr)
        loc = torch.zeros(b_shape)
        var = torch.ones(b_shape)
        self.b_dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        
        #estimate scalar fudge factor to make unit variance.
        self.rescale = 1.
        samples = self.sample_n(10000)
        self.rescale = 1./np.std(utilities.grab(samples))
                       
    def get_lxly(self, nx, dx):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi ) 

    def get_ell(self,nx, dx):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
        lx, ly = self.get_lxly(nx, dx)
        return np.sqrt(lx**2 + ly**2)    
        
    def log_prob(self, x):
        #ignore constant factors
        
        #fft to get the modes
        fft = torch.fft.rfftn(x,dim=[1,2]) * np.sqrt(2.)
        fft[:] *= self.clinvfactor / self.rescale
        x = torch.view_as_real(fft)  
        
        #naive: ignore symmetries
        #logp = self.dist.log_prob(x.reshape(x.shape[0], -1)) 
        #logp = torch.sum(logp, dim=1)
        
        #correct: use symmetries
        a = x[:,:,:,0]
        b = x[:,:,:,1]
        amasked = a[:,self.a_mask]
        bmasked = b[:,self.b_mask]
        logp_a = self.a_dist.log_prob(amasked) 
        logp_b = self.b_dist.log_prob(bmasked)
        logp = torch.sum(logp_a, dim=1) + torch.sum(logp_b, dim=1)
        
        return logp
    
    def sample_n(self, batch_size):
        #https://pytorch.org/docs/stable/complex_numbers.html
        
        #draw random rfft modes
        x = self.dist.sample((batch_size,))
        
        #test logp
        #logptemp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        #print("logp temp", torch.sum(logptemp, dim=1))
        
        #reshape to rfft format
        x = x.reshape(batch_size, *self.rfourier_shape)
        #make complex data type
        fft = torch.view_as_complex(x) / np.sqrt(2.)
         
        #enforce rfft constraints
        #from quicklens
        fft[:,0,0] = np.sqrt(2.) * fft[:,0,0].real #fft.real
        fft[:,int(self.nx/2+1):, 0] = torch.conj( torch.flip(fft[:,1:int(self.nx/2),0], (1,)) ) 
        
        #extra symmetries (assuming th rfft output format is as in numpy)
        fft[:,0,int(self.nx/2)] = fft[:,0,int(self.nx/2)].real * np.sqrt(2.)
        fft[:,int(self.nx/2),0] = fft[:,int(self.nx/2),0].real * np.sqrt(2.)
        fft[:,int(self.nx/2),int(self.nx/2)] = fft[:,int(self.nx/2),int(self.nx/2)].real * np.sqrt(2.)
        fft[:,int(self.nx/2+1):, int(self.nx/2)] = torch.conj( torch.flip(fft[:,1:int(self.nx/2),int(self.nx/2)], (1,)) ) 
        #flip from https://github.com/pytorch/pytorch/issues/229
        #https://pytorch.org/docs/stable/generated/torch.flip.html#torch.flip
        
        #TODO: check normalization of irfftn. see quicklens maps.py line 907 and the new options of irfftn.
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        #for now scalar fudge factor to make unit variance.
        
        #adjust mode amplitude to power spectrum
        fft[:] *= self.clfactor * self.rescale
        
        #transform to position space
        rmap = torch.fft.irfftn(fft,dim=[1,2])
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        
        return rmap
    


class CorrelatedNormalNew:
    def __init__(self, loc, var, nx, dx, cl_theo_in, torch_device):
        self.torch_device=torch_device
        self.nx=nx
        self.dx=dx
        self.cl_theo=np.ones((len(cl_theo_in)))
        self.cl_theo[:]=cl_theo_in[:]
        
        #normal distribution to draw random fourier modes
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.rfourier_shape = loc.shape
        
        #create the array to multiply the fft with to get the desired power spectrum
        self.cl_theo[0:1] = self.cl_theo[2]
        
        self.ells_flat = self.get_ell(self.nx, self.dx).flatten()
        
        #self.interp_ells_flat = scipy.interpolate.interp1d(np.log(self.ells_flat[1:]), np.log(cl_theo[1:]), 
        #                                                   kind='linear', fill_value=0, bounds_error=False)
        
        interp_cl = np.interp(self.ells_flat[1:], np.arange(0, len(self.cl_theo[1:])), np.sqrt(self.cl_theo[1:]), right=0)
        
        clfactor = np.zeros((self.ells_flat.shape))
        
        clfactor[1:] = interp_cl[:]
        clfactor[0] = clfactor[1]
        
        clfactor = clfactor.reshape(self.rfourier_shape[0:2])
        
        self.clfactor_np = clfactor
        
        self.clfactor = torch.from_numpy(clfactor).float().to(torch_device)
        clinvfactor = np.copy(clfactor) 
        #clinvfactor[clinvfactor==0] = 1. #TODO: should we to remove the monopole?
        clinvfactor = 1./clinvfactor
        self.clinvfactor = torch.from_numpy(clinvfactor).float().to(torch_device)
        
        #masks for rfft symmetries
        a_mask = np.ones((self.nx, int(self.nx/2+1)), dtype=bool)
        #a_mask[0, 0] = False #MM added ####################### commented
        a_mask[int(self.nx/2+1):, 0] = False
        a_mask[int(self.nx/2+1):, int(nx/2)] = False
        b_mask = np.ones((self.nx, int(self.nx/2+1)), dtype=bool)
        b_mask[0, 0] = False
        b_mask[0, int(self.nx/2)] = False
        b_mask[int(self.nx/2), 0] = False
        b_mask[int(self.nx/2), int(self.nx/2)] = False
        b_mask[int(self.nx/2+1):, 0] = False
        b_mask[int(self.nx/2+1):, int(self.nx/2)] = False
        self.a_mask = a_mask
        self.b_mask = b_mask
        
        #how many mask elements
        a_nr = self.a_mask.sum()
        b_nr = self.b_mask.sum()
        #print (a_nr,b_nr)

        #make distributions with the right number of elements for each re and im mode.
        a_shape = (a_nr)
        loc = torch.zeros(a_shape)
        var = torch.ones(a_shape)
        self.a_dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        b_shape = (b_nr)
        loc = torch.zeros(b_shape)
        var = torch.ones(b_shape)
        self.b_dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        
        #estimate scalar fudge factor to make unit variance.
        self.rescale = 1.
        samples = self.sample_n(1000)
        self.rescale = 1./np.std(utilities.grab(samples))
        del samples
        torch.cuda.empty_cache()
                       
    def get_lxly(self, nx, dx):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid(np.fft.fftfreq(nx, dx)[0:int(nx/2+1)]*2.*np.pi, np.fft.fftfreq(nx, dx)*2.*np.pi)

    def get_ell(self,nx, dx):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
        lx, ly = self.get_lxly(nx, dx)
        return np.sqrt(lx**2 + ly**2)
        
    def log_prob(self, x):
        #ignore constant factors
        
        #fft to get the modes
        fft = torch.fft.rfftn(x, dim=[1, 2]) * np.sqrt(2.)
        fft[:] *= self.clinvfactor / self.rescale
        x = torch.view_as_real(fft)
        
        #naive: ignore symmetries
        #logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        #logp = torch.sum(logp, dim=1)
        
        #correct: use symmetries
        a = x[:, :, :, 0]
        b = x[:, :, :, 1]
        amasked = a[:, self.a_mask]
        bmasked = b[:, self.b_mask]
        logp_a = self.a_dist.log_prob(amasked)
        logp_b = self.b_dist.log_prob(bmasked)
        logp = torch.sum(logp_a, dim=1) + torch.sum(logp_b, dim=1)
        
        return logp
    
    def sample_n(self, batch_size):
        #https://pytorch.org/docs/stable/complex_numbers.html

        #draw random rfft modes
        x = self.dist.sample((batch_size,))
        
        for map_id in range(batch_size):
            x_mean = torch.mean(x[map_id])
            x_std = torch.std(x[map_id])
            #x[map_id] = (x[map_id] - x_mean) / x_std
            x[map_id] = x[map_id] / x_std

        #test logp
        logptemp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        #print("logp temp", torch.sum(logptemp, dim=1))        

        #reshape to rfft format
        x = x.reshape(batch_size, *self.rfourier_shape)
        #make complex data type
        fft = torch.view_as_complex(x) / np.sqrt(2.)
        
        a = x[:, :, :, 0]
        b = x[:, :, :, 1]
        amasked = a[:, self.a_mask]
        bmasked = b[:, self.b_mask]
        logp_a = self.a_dist.log_prob(amasked)
        logp_b = self.b_dist.log_prob(bmasked)

        #enforce rfft constraints
        #from quicklens
        #fft[:, 0, 0] = 0 #MM replaced from: np.sqrt(2.) * fft[:,0,0].real 
        fft[:, 0, 0] = np.sqrt(2.) * fft[:, 0, 0].real 
        fft[:, int(self.nx/2+1):, 0] = torch.conj(torch.flip(fft[:, 1:int(self.nx/2), 0], (1,))) 

        #extra symmetries (assuming th rfft output format is as in numpy)
        fft[:, 0, int(self.nx/2)] = fft[:, 0, int(self.nx/2)].real * np.sqrt(2.)
        fft[:, int(self.nx/2), 0] = fft[:, int(self.nx/2), 0].real * np.sqrt(2.)
        fft[:, int(self.nx/2), int(self.nx/2)] = fft[:, int(self.nx/2), int(self.nx/2)].real * np.sqrt(2.)
        fft[:, int(self.nx/2+1):, int(self.nx/2)] = torch.conj(torch.flip(fft[:, 1:int(self.nx/2), int(self.nx/2)], (1,)))
        #flip from https://github.com/pytorch/pytorch/issues/229
        #https://pytorch.org/docs/stable/generated/torch.flip.html#torch.flip

        #TODO: check normalization of irfftn. see quicklens maps.py line 907 and the new options of irfftn.
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        #for now scalar fudge factor to make unit variance.

        #adjust mode amplitude to power spectrum
        fft[:] *= self.clfactor * self.rescale

        #transform to position space
        rmap = torch.fft.irfftn(fft, dim=[1, 2])
        #https://pytorch.org/docs/1.7.1/fft.html#torch.fft.irfftn
        
        return rmap


######################## Flow generalities


def apply_flow_to_prior(prior, coupling_layers, batch_size):
    #draws from the prior (base distribution) and flows them
    u = prior.sample_n(batch_size)
    #u = prior_tensor
    log_pu = prior.log_prob(u)
    z = u.clone()
    log_pz = log_pu.clone()
    for layer in coupling_layers:
        z, logJ = layer.forward(z)
        log_pz = log_pz - logJ
    return u, log_pu, z, log_pz


def apply_flow_to_prior_maps(u, coupling_layers, batch_size):
    #draws from the prior (base distribution) and flows them
    #u = prior_tensor
    log_pu = prior.log_prob(u)
    z = u.clone()
    log_pz = log_pu.clone()
    for layer in coupling_layers:
        z, logJ = layer.forward(z)
        log_pz = log_pz - logJ
    return u, log_pu, z, log_pz


def apply_reverse_flow_to_sample(z, prior, coupling_layers):
    #takes samples and calculates their representation in base distribution space and their density 
    log_J_Tinv = 0
    n_layers = len(coupling_layers)
    for layer_id in reversed(range(n_layers)):
        layer = coupling_layers[layer_id]
        z, logJ = layer.reverse(z)
        log_J_Tinv = log_J_Tinv + logJ 
    u = z
    log_pu = prior.log_prob(u)
    return u, log_pu, log_J_Tinv


def apply_reverse_flow_to_sample_BASEDISTRIBUTION(z, prior, coupling_layers):
    log_pu = prior.log_prob(z)
    return z, log_pu, 1.
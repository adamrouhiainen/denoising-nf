import numpy as np
import scipy
import torch

import flow_architecture



def get_lxly(nx, dx, ny=None, dy=None):
    """ Returns the (lx, ly) pair associated with each Fourier mode """
    if ny is None:
        ny = nx
        dy = dx
    return np.meshgrid(np.fft.fftfreq(nx, dx)[0:nx//2+1]*2*np.pi, np.fft.fftfreq(ny, dy)*2*np.pi)


def get_ell(nx, dx, ny=None, dy=None):
    """ Returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
    lx, ly = get_lxly(nx, dx)
    return np.sqrt(lx**2 + ly**2)


def get_rfft(rmap, nx, dx):
    """ Return an rfft array containing the real Fourier transform of this map """
    #From quicklens.maps.rmap.get_rfft()
    tfac = dx / nx
    rfft = torch.fft.rfft2(rmap) * tfac
    return rfft



class Powerspectra():
    def __init__(self, params, cl_theo_ell=None, cl_theo=None, cl_max=None):
        self.params = params
        
        if params.use_ql:
            """self.ell = get_ell(self.params.nx, self.params.dx).flatten()
            
            self.qlfunc = ql.spec.get_camb_lensedcl(lmax=self.params.lmax)
            self.cl_ql = self.qlfunc.cltt / params.norm_fac**2 #this factor is to make the cl same size as the flow samples
            self.ell_ql = np.arange(0, self.qlfunc.cltt.shape[0])
            cl_tt = self.cl_ql
            self.cl_tt_no_interpolate = cl_tt
            cl_tt[0:2] = cl_tt[2]
            interp = scipy.interpolate.interp1d(np.log(self.ell_ql[1:]), np.log(cl_tt[1:]), kind='linear', fill_value=0, bounds_error=False)
            self.cl_tt = np.zeros(self.ell.shape[0])
            self.cl_tt[1:] = np.exp(interp(np.log(self.ell[1:])))
            self.cl_tt[0] = self.cl_tt[1]"""
        else:
            self.ell = get_ell(self.params.nx, self.params.dx).flatten()
            
            self.cl_theo_ell = cl_theo_ell
            self.cl_theo = cl_theo
            self.cl_theo[0:2] = cl_theo[2]
            
            if cl_max is None:
                interp = scipy.interpolate.interp1d(self.cl_theo_ell[1:], cl_theo[1:], kind='linear', fill_value=0, bounds_error=False)
                self.cl_tt = np.zeros(self.ell.shape[0])
                self.cl_tt[1:] = interp(self.ell[1:])
                self.cl_tt[0] = self.cl_tt[1]
            else:
                interp = scipy.interpolate.interp1d(self.cl_theo_ell[:cl_max], cl_theo[:cl_max], kind='linear', fill_value=0, bounds_error=False)
                self.cl_tt = np.zeros(cl_max)
                self.cl_tt[:] = interp(self.ell[:cl_max])
            
            #interp = scipy.interpolate.interp1d(np.log(self.cl_theo_ell[:cl_len]), np.log(cl_theo[:cl_len]), kind='linear', fill_value=0, bounds_error=False)
            #self.cl_tt = np.zeros(self.ell.shape[0])
            #self.cl_tt[:] = np.exp(interp(self.ell[:cl_len]))
        
        
    def inverse_ps_weight(self, data):
        """ Divides data by cls; if len(data) is smaller than len(cl), then cl list is stretched to match data """
        cl = self.cl_tt
        cl = torch.tensor(cl)
        
        if data.shape[0] < cl.shape[0]:
            skip = cl.shape[0]//data.shape[1]
            cl = cl[0::skip]
            cl = cl[:data.shape[1]]
            
        data = torch.div(data, cl)
        return data
    


class Lossfunctions():
    def __init__(self, params, cl_theo_ell=None, cl_theo=None, cl_max=None):
        self.params = params
        self.powerspectra = Powerspectra(self.params, cl_theo_ell=cl_theo_ell, cl_theo=cl_theo, cl_max=cl_max)
        
        
    def fourier_loss_auto(self, y, return_power=False):
        """ Computes the prior of torch.tensor with the Powerspectra Class """
        rfft = get_rfft(y, self.params.nx, self.params.dx)
        rfft_shape = rfft.size()

        power_array = torch.real((rfft*torch.conj(rfft)))
        power = torch.reshape(power_array, [-1, rfft_shape[1]*rfft_shape[2]])
        power = self.powerspectra.inverse_ps_weight(power)
        
        loss = torch.sum(power)
        if not return_power: return loss
        else: return loss, power
    
    
    def realspace_loss_noisediag(self, y_true, y_pred):
        """ Computes the likelihood of two torch.tensors, excluding the masked region """
        loss = (y_pred-y_true)**2 / self.params.noise_pix
        loss = loss*torch.tensor(self.params.mask)
        if   len(loss.shape)==3: loss = torch.sum(loss, dim=[1, 2])
        elif len(loss.shape)==4: loss = torch.sum(loss, dim=[1, 2, 3])
        return loss
    
    
    def realspace_loss_noisediag_patching(self, y_true, y_pred, patch_id):
        """ Computes the likelihood of two torch.tensors, excluding the masked region matching the patch_id for each patch """
        loss = (y_pred-y_true)**2 / self.params.noise_pix
        #loss = loss*torch.tensor(self.params.mask_patch[patch_id])
        loss = loss*self.params.mask_patches[patch_id]
        if   len(loss.shape)==3: loss = torch.sum(loss, dim=[1, 2])
        elif len(loss.shape)==4: loss = torch.sum(loss, dim=[1, 2, 3])
        return loss
    
    
    def loss_wiener_J3(self, y_true, y_pred):
        """ Computes the posterior of two torch.tensors """
        term1 = self.realspace_loss_noisediag(y_true, y_pred)
        term2 = self.fourier_loss_auto(y_pred)
        return term1, term2
    
    
    def loss_wiener_J3_flow(self, y_true, y_pred, prior, layers):
        """ Computes the posterior of two torch.tensors using a pre-trained Real NVP to compute the prior """
        term1 = self.realspace_loss_noisediag(y_true, y_pred)
        
        _, log_pu, log_J_Tinv = flow_architecture.apply_reverse_flow_to_sample(y_pred, prior, layers)
        term2 = -(log_pu + log_J_Tinv)
        
        return term1, term2
    
    
    def loss_wiener_J3_flow_patching(self, y_true, y_pred, prior, layers, patch_id):
        """ Computes the posterior of two torch.tensors using a pre-trained Real NVP to compute the prior, with patching to upscale """
        term1 = self.realspace_loss_noisediag_patching(y_true, y_pred, patch_id)
        
        _, log_pu, log_J_Tinv = flow_architecture.apply_reverse_flow_to_sample(y_pred, prior, layers)
        term2 = -(log_pu + log_J_Tinv)
        
        return term1, term2
    

    def loss_wiener_J3_Glow(self, y_true, y_pred, model):
        """ Computes the posterior of two torch.tensors using a pre-trained Glow to compute the prior """
        term1 = self.realspace_loss_noisediag(y_true, y_pred)

        log_pu, log_J_Tinv, _ = model(y_pred)
        term2 = -log_pu + log_J_Tinv

        return term1, term2
    
    
    def loss_J2(self, y_true, y_pred):
        return 0.5*torch.sum((y_true - y_pred)**2)
    
    
    def loss_J2_rfft(self, y_true, y_pred):
        return torch.real(0.5*torch.sum((torch.fft.rfft(y_true) - torch.fft.rfft(y_pred))**2))
    
    
    def loss_cross_corr_rfft(self, y_true, y_pred):
        """ Computes the pixel-wise cross-correlation coefficient of two torch.tensors in fourier space """
        numerator = torch.sum(torch.fft.rfft(y_true) * torch.fft.rfft(y_pred))
        denominator = torch.sqrt(torch.sum(torch.fft.rfft(y_true)**2) * torch.sum(torch.fft.rfft(y_pred)**2))
        return torch.real(numerator / denominator)
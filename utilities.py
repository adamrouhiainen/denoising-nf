import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import flow_architecture

r2d = 180./np.pi
d2r = np.pi/180.
rad2arcmin = 180.*60./np.pi


def grab(var):
  return var.detach().cpu().numpy()


def plot_lists(list_1=None, list_2=[], idmin=0, trunc=0, ymin=None, ymax=None, offset=0, figsize=(5, 3), ylog=False, label1='', label2='', xlabel='', ylabel='', title='', file_name=None):
    idmax = len(list_1) - trunc
    fig=plt.figure(figsize=figsize)
    plt.plot(np.arange(idmin, idmax-offset, 1), list_1[idmin:idmax-offset], color='red', label=label1)
    if len(list_2) > 0: plt.plot(np.arange(idmin, idmax-offset, 1), list_2[idmin:idmax], label=label2)
    if not label1=='': plt.legend(loc=1, frameon=False, fontsize=14)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title)
    if ylog: ax.set_yscale('log')
    if not (ymin==None and ymax==None): plt.ylim([ymin, ymax])
    fig.tight_layout()
    if not file_name==None: plt.savefig(file_name)
    plt.show()
    

def imshow(array, vmin=None, vmax=None, figsize=(8, 8), title='', axis=True, colorbar=True, file_name=None):
    plt.figure(figsize=figsize)
    if (vmin==None and vmax==None): plt.imshow(array)
    else: plt.imshow(array, vmin=vmin, vmax=vmax)
    if not axis: plt.axis('off')
    if colorbar: plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    if not file_name==None: plt.savefig(file_name)
    
    
##################### patching maps


def make_small_maps_from_big_map(big_map, n, displace=None):
    """
    Makes small maps of shape ((N//n)**2, n, n) from big_map
    big_map must be periodic
    
    big_map: torch.tensor of shape (N, N)
    n: Length of small_map; assumes N%n == 0
    displace: If True, big_map is shifted by (n//2, n//2) pixels
    """
    N = big_map.shape[-1]
    small_maps = torch.zeros(((N//n)**2, n, n))
    
    if   displace == 1: big_map = torch.roll(big_map, n//2, dims=-2)
    elif displace == 2: big_map = torch.roll(big_map, n//2, dims=-1)
    elif displace == 3: big_map = torch.roll(big_map, (n//2, n//2), dims=(-2, -1))
    
    for i in range(N//n):
        for j in range(N//n):
            small_maps[i+(N//n)*j] = big_map[i*n:(i+1)*n, j*n:(j+1)*n]
            
    return small_maps


def make_big_map_from_small_maps(small_maps_0, small_maps_1, small_maps_2, small_maps_3, N):
    """
    Makes big_map of shape(N, N) from small maps of shape ((N//n)**2, n, n)
    
    small_maps_0: small maps
    small_maps_1: small maps displaced by n//2
    small_maps_2: small maps displaced by n//2 in the other dimension
    small_maps_3: small maps displaced by (n//2, n//2)
    N: length of big_map
    """
    n = small_maps_0.shape[-1]
    d = n//4
    
    #Shorten edges of small maps
    pad = (-d, -d, -d, -d)
    small_maps_0 = F.pad(small_maps_0, pad)
    small_maps_1 = F.pad(small_maps_1, pad)
    small_maps_2 = F.pad(small_maps_2, pad)
    small_maps_3 = F.pad(small_maps_3, pad)
    
    big_map = torch.zeros((N, N))
    
    for i in range(N//n):
        for j in range(N//n):
            big_map[i*n+d:i*n+d+n//2, j*n+d:j*n+d+n//2] = small_maps_0[i+(N//n)*j]
            
    #Roll big_map to get to small_maps_1 position
    big_map = torch.roll(big_map, n//2, dims=-2)
    for i in range(N//n):
        for j in range(N//n):
            big_map[i*n+d:i*n+d+n//2, j*n+d:j*n+d+n//2] = small_maps_1[i+(N//n)*j]
            
    #This roll is now diagonal from the original big_map, so use small_maps_3
    big_map = torch.roll(big_map, n//2, dims=-1)
    for i in range(N//n):
        for j in range(N//n):
            big_map[i*n+d:i*n+d+n//2, j*n+d:j*n+d+n//2] = small_maps_3[i+(N//n)*j]
            
    #Roll backwards to get to small_maps_2
    big_map = torch.roll(big_map, -n//2, dims=-2)
    for i in range(N//n):
        for j in range(N//n):
            big_map[i*n+d:i*n+d+n//2, j*n+d:j*n+d+n//2] = small_maps_2[i+(N//n)*j]
            
    #Go back to original big_map position
    big_map = torch.roll(big_map, -n//2, dims=-1)
    
    return big_map


def low_of_fft(tensor, lr_factor, mode='l2_norm'):
    #Sets large modes to 0 of a (B, L, W) numpy tensor. Input tensor comes from np.fft.fft2.
    B = tensor.shape[0]
    l_x = tensor.shape[-2]
    l_y = tensor.shape[-1]
    output = np.zeros((B, l_x, l_y)).astype(complex)
    for i in range(l_x):
        for j in range(l_y):
            if mode == 'l2_norm':
                if np.sqrt(i**2+j**2)<=(l_x/lr_factor):
                    output[:, i, j] = tensor[:, i, j]
                    output[:, i, l_y-j-1] = tensor[:, i, l_y-j-1]
                    output[:, l_x-i-1, j] = tensor[:, l_x-i-1, j]
                    output[:, l_x-i-1, l_y-j-1] = tensor[:, l_x-i-1, l_y-j-1]
    return output

def high_of_fft(tensor, lr_factor, mode='l2_norm'):
    #Sets large modes to 0 of a (B, L, W) numpy tensor. Input tensor comes from np.fft.fft2.
    B = tensor.shape[0]
    l_x = tensor.shape[-2]
    l_y = tensor.shape[-1]
    output = tensor.astype(complex)
    for i in range(l_x):
        for j in range(l_y):
            if mode == 'l2_norm':
                if np.sqrt(i**2+j**2)<(l_x/lr_factor):
                    output[:, i, j] = 0.
                    output[:, i, l_y-j-1] = 0.
                    output[:, l_x-i-1, j] = 0.
                    output[:, l_x-i-1, l_y-j-1] = 0.
    return output
    


def get_lxly(nx, dx):
    #returns the (lx, ly) pair associated with each Fourier mode
    return np.meshgrid(np.fft.fftfreq(nx, dx)*2.*np.pi,np.fft.fftfreq(nx, dx)*2.*np.pi)


def get_ell_squared(nx, dx):
    #returns the wavenumber l = lx**2 + ly**2 for each Fourier mode
    lx, ly = get_lxly(nx, dx)
    return lx**2 + ly**2


def add_noise(array, std=1.):
    """ Adds std noise per pixel to a 2d np.ndarray """
    batch_size = np.shape(array)[0]
    L = np.shape(array)[1]
    W = np.shape(array)[2]
    noise_prior = flow_architecture.SimpleNormal(torch.zeros((L, W)), torch.ones((L, W))*std)
    noise = grab(noise_prior.sample_n(batch_size))
    return array + noise
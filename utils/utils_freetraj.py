import torch
import torch.fft as fft
import math


def get_longpath(BOX_SIZE_H=0.3, BOX_SIZE_W=0.3, input_mode=4):

    if input_mode == 1:
        # mode 1
        inputs = [[0, 0, 0 + BOX_SIZE_H, 0, 0 + BOX_SIZE_W], 
                [7, 1-BOX_SIZE_H, 1, (1-BOX_SIZE_W) / 15 * 7, (1-BOX_SIZE_W) / 15 * 7 + BOX_SIZE_W], 
                [8, 1-BOX_SIZE_H, 1, (1-BOX_SIZE_W) / 15 * 8, (1-BOX_SIZE_W) / 15 * 8 + BOX_SIZE_W], 
                [15, 0, 0 + BOX_SIZE_H, 1-BOX_SIZE_W, 1],
                [16, 0.1, 0.1 + BOX_SIZE_H, 0.9-BOX_SIZE_W, 0.9],
                [25, 0.1, 0.1 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W],
                [31, 0.9-BOX_SIZE_H, 0.9, 0.1, 0.1 + BOX_SIZE_W],
                [32, 1-BOX_SIZE_H, 1, 0, 0 + BOX_SIZE_W],
                [39, 0, 0 + BOX_SIZE_H, (1-BOX_SIZE_W) / 15 * 7, (1-BOX_SIZE_W) / 15 * 7 + BOX_SIZE_W],
                [40, 0, 0 + BOX_SIZE_H, (1-BOX_SIZE_W) / 15 * 8, (1-BOX_SIZE_W) / 15 * 8 + BOX_SIZE_W],
                [47, 1-BOX_SIZE_H, 1, 1-BOX_SIZE_W, 1],
                [48, 0.9-BOX_SIZE_H, 0.9, 0.9-BOX_SIZE_W, 0.9],
                [57, 0.9-BOX_SIZE_H, 0.9, 0.1, 0.1 + BOX_SIZE_W],
                [63, 0.1, 0.1 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W]]
    elif input_mode == 2:
        # mode 2
        inputs = [[0, 0.1, 0.1 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W], 
                  [6, 0.9-BOX_SIZE_H, 0.9, 0.1, 0.1 + BOX_SIZE_W], 
                  [15, 0.9-BOX_SIZE_H, 0.9, 0.9-BOX_SIZE_W, 0.9],
                  [16, 0.9-BOX_SIZE_H, 0.9, 0.9-BOX_SIZE_W, 0.9], 
                  [22, 0.1, 0.1 + BOX_SIZE_H, 0.9-BOX_SIZE_W, 0.9], 
                  [31, 0.1, 0.1 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W],
                  [32, 0.1, 0.1 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W],
                  [41, 0.1, 0.1 + BOX_SIZE_H, 0.9-BOX_SIZE_W, 0.9],
                  [47, 0.9-BOX_SIZE_H, 0.9, 0.9-BOX_SIZE_W, 0.9],
                  [48, 0.9-BOX_SIZE_H, 0.9, 0.9-BOX_SIZE_W, 0.9],
                  [57, 0.9-BOX_SIZE_H, 0.9, 0.1, 0.1 + BOX_SIZE_W],
                  [63, 0.1, 0.1 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W]]
    elif input_mode == 3:
        # mode 3 ||||
        inputs = [[0, 0, 0 + BOX_SIZE_H, 0, 0 + BOX_SIZE_W],
            [9, 1-BOX_SIZE_H, 1, (1-BOX_SIZE_W) / 7 * 1, (1-BOX_SIZE_W) / 7 * 1 + BOX_SIZE_W],
            [18, 0, 0 + BOX_SIZE_H, (1-BOX_SIZE_W) / 7 * 2, (1-BOX_SIZE_W) / 7 * 2 + BOX_SIZE_W],
            [27, 1-BOX_SIZE_H, 1, (1-BOX_SIZE_W) / 7 * 3, (1-BOX_SIZE_W) / 7 * 3 + BOX_SIZE_W],
            [36, 0, 0 + BOX_SIZE_H, (1-BOX_SIZE_W) / 7 * 4, (1-BOX_SIZE_W) / 7 * 4 + BOX_SIZE_W],
            [45, 1-BOX_SIZE_H, 1, (1-BOX_SIZE_W) / 7 * 5, (1-BOX_SIZE_W) / 7 * 5 + BOX_SIZE_W],
            [54, 0, 0 + BOX_SIZE_H, (1-BOX_SIZE_W) / 7 * 6, (1-BOX_SIZE_W) / 7 * 6 + BOX_SIZE_W],
            [63, 1-BOX_SIZE_H, 1, 1-BOX_SIZE_W, 1]]
    elif input_mode == 4:
        # mode 4 ----
        inputs = [[0, 0, 0 + BOX_SIZE_H, 0, 0 + BOX_SIZE_W],
            [9, (1-BOX_SIZE_H) / 7 * 1, (1-BOX_SIZE_H) / 7 * 1 + BOX_SIZE_H, 1-BOX_SIZE_W, 1],
            [18, (1-BOX_SIZE_H) / 7 * 2, (1-BOX_SIZE_H) / 7 * 2 + BOX_SIZE_H, 0, 0 + BOX_SIZE_W],
            [27, (1-BOX_SIZE_H) / 7 * 3, (1-BOX_SIZE_H) / 7 * 3 + BOX_SIZE_H, 1-BOX_SIZE_W, 1],
            [36, (1-BOX_SIZE_H) / 7 * 4, (1-BOX_SIZE_H) / 7 * 4 + BOX_SIZE_H, 0, 0 + BOX_SIZE_W],
            [45, (1-BOX_SIZE_H) / 7 * 5, (1-BOX_SIZE_H) / 7 * 5 + BOX_SIZE_H, 1-BOX_SIZE_W, 1],
            [54, (1-BOX_SIZE_H) / 7 * 6, (1-BOX_SIZE_H) / 7 * 6 + BOX_SIZE_H, 0, 0 + BOX_SIZE_W],
            [63, 1-BOX_SIZE_H, 1, 1-BOX_SIZE_W, 1]]
    else:
        print('error')
        exit()

    outputs = plan_path(inputs)
    # print(outputs)
    return outputs

def get_path(BOX_SIZE_H=0.3, BOX_SIZE_W=0.3, input_mode=0):

    if input_mode == 0:
        # \ d
        inputs = [[0, 0, 0 + BOX_SIZE_H, 0, 0 + BOX_SIZE_W], [15, 1-BOX_SIZE_H, 1, 1-BOX_SIZE_W, 1]] 
    elif input_mode == 1:
        # / re d
        inputs = [[0, 0, 0 + BOX_SIZE_H, 1-BOX_SIZE_W, 1], [15, 1-BOX_SIZE_H, 1, 0, 0 + BOX_SIZE_W]] 
    elif input_mode == 2:        
        # L
        inputs = [[0, 0.1, 0.1 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W], [6, 0.9-BOX_SIZE_H, 0.9, 0.1, 0.1 + BOX_SIZE_W], [15, 0.9-BOX_SIZE_H, 0.9, 0.9-BOX_SIZE_W, 0.9]] 
    elif input_mode == 3:     
        # re L
        inputs = [[0, 0.9-BOX_SIZE_H, 0.9, 0.9-BOX_SIZE_W, 0.9], [6, 0.1, 0.1 + BOX_SIZE_H, 0.9-BOX_SIZE_W, 0.9], [15, 0.1, 0.1 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W]] 
    elif input_mode == 4:     
        # V
        inputs = [[0, 0, 0 + BOX_SIZE_H, 0, 0 + BOX_SIZE_W], [7, 1-BOX_SIZE_H, 1, (1-BOX_SIZE_W) / 15 * 7, (1-BOX_SIZE_W) / 15 * 7 + BOX_SIZE_W], [8, 1-BOX_SIZE_H, 1, (1-BOX_SIZE_W) / 15 * 8, (1-BOX_SIZE_W) / 15 * 8 + BOX_SIZE_W], [15, 0, 0 + BOX_SIZE_H, 1-BOX_SIZE_W, 1]]
    elif input_mode == 5:     
        # re V
        inputs = [[0, 1-BOX_SIZE_H, 1, 1-BOX_SIZE_W, 1], [7, 0, 0 + BOX_SIZE_H, (1-BOX_SIZE_W) / 15 * 8, (1-BOX_SIZE_W) / 15 * 8 + BOX_SIZE_W], [8, 0, 0 + BOX_SIZE_H, (1-BOX_SIZE_W) / 15 * 7, (1-BOX_SIZE_W) / 15 * 7 + BOX_SIZE_W], [15, 1-BOX_SIZE_H, 1, 0, 0 + BOX_SIZE_W]]
    elif input_mode == 6:    
        # -- goback
        inputs = [[0, 0.35, 0.35 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W], [7, 0.35, 0.35 + BOX_SIZE_H, 0.9-BOX_SIZE_W, 0.9], [8, 0.35, 0.35 + BOX_SIZE_H, 0.9-BOX_SIZE_W, 0.9], [15, 0.35, 0.35 + BOX_SIZE_H, 0.1, 0.1 + BOX_SIZE_W]] 
    elif input_mode == 7:    
        # tri
        inputs = [[0, 0.1, 0.1 + BOX_SIZE_H, 0.35, 0.35 + BOX_SIZE_W], [5, 0.9-BOX_SIZE_H, 0.9, 0.9-BOX_SIZE_W, 0.9], [10, 0.9-BOX_SIZE_H, 0.9, 0.1, 0.1 + BOX_SIZE_W], [15, 0.1, 0.1 + BOX_SIZE_H, 0.35, 0.35 + BOX_SIZE_W]]

    outputs = plan_path(inputs)
    return outputs

# input: List([frame, h_start, h_end, w_start, w_end], ...)
# return: List([h_start, h_end, w_start, w_end], ...)
def plan_path(input):
    len_input = len(input)
    path = [input[0][1:]]
    for i in range(1, len_input):
        start = input[i-1]
        end = input[i]
        start_frame = start[0]
        end_frame = end[0]
        h_start_change = (end[1] - start[1]) / (end_frame - start_frame)
        h_end_change = (end[2] - start[2]) / (end_frame - start_frame)
        w_start_change = (end[3] - start[3]) / (end_frame - start_frame)
        w_end_change = (end[4] - start[4]) / (end_frame - start_frame)
        for j in range(start_frame+1, end_frame + 1):
            increase_frame = j - start_frame
            path += [[increase_frame * h_start_change + start[1], increase_frame * h_end_change + start[2], increase_frame * w_start_change + start[3], increase_frame * w_end_change + start[4]]]
 
    return path


def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """ 2d Gaussian weight function
    """
    gaussian_map = (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )
    gaussian_map.div_(gaussian_map.max())
    return gaussian_map

def gaussian_weight(height=32, width=32, KERNEL_DIVISION=3.0):

    x = torch.linspace(0, height, height)
    y = torch.linspace(0, width, width)
    x, y = torch.meshgrid(x, y, indexing="ij")
    noise_patch = (
                    gaussian_2d(
                        x,
                        y,
                        mx=int(height / 2),
                        my=int(width / 2),
                        sx=float(height / KERNEL_DIVISION),
                        sy=float(width / KERNEL_DIVISION),
                    )
                ).half()
    return noise_patch

def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed


def get_freq_filter(shape, device, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask

def load_idx(prompt_file):
    f = open(prompt_file, 'r')
    idx_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            indices = l.split(',')
            indices_list = []
            for index in indices:
                indices_list.append(int(index))
            idx_list.append(indices_list)
        f.close()
    return idx_list

def load_traj(prompt_file):
    f = open(prompt_file, 'r')
    traj_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            numbers = l.split(',')
            numbers_list = []
            for number_index in range(len(numbers)):
                if number_index == 0:
                    numbers_list.append(int(numbers[number_index]))
                else:
                    numbers_list.append(float(numbers[number_index]))
            traj_list.append(numbers_list)
        f.close()
    return traj_list
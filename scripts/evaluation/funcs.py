#import sys
from collections import OrderedDict
import torch
#sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from einops import rearrange
from safetensors.torch import load_file

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    pass

def load_model_checkpoint(model, file_path, dtype, device):
    if "safetensors" in file_path:
        try:
            state_dict = load_file(file_path)
        except:
            state_dict = torch.load(file_path, map_location="cpu")
    else:
        state_dict = torch.load(file_path, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]

    filtered_state_dict = {
        k: v 
        for k, v in state_dict.items() 
        if not (k.startswith("cond_stage_model") or k.startswith("embedder"))
        #if not (k.startswith("cond_stage_model"))
    }  # Filter out keys starting with "cond_stage_model" and "embedder"
    if is_accelerate_available:
        for key in filtered_state_dict:
            set_module_tensor_to_device(model, key, dtype=dtype, device=device, value=filtered_state_dict[key])
    else:
        model.load_state_dict(filtered_state_dict, strict=True)

    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

def get_latent_z_with_hidden_states(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)

    hidden_states_first_last = []
    ### use only the first and last hidden states
    for hid in hidden_states:
        hid = rearrange(hid, '(b t) c h w -> b c t h w', t=t)
        hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
        hidden_states_first_last.append(hid_new)

    z = model.get_first_stage_encoding(encoder_posterior).detach()
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z, hidden_states_first_last
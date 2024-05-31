#import sys
from collections import OrderedDict
import torch
#sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from einops import rearrange
from safetensors.torch import load_file

def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        if "safetensors" in ckpt:
            try:
                state_dict = load_file(ckpt)
            except:
                state_dict = torch.load(ckpt, map_location="cpu")
        else:
            state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=full_strict)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        # else:
        #     ## deepspeed
        #     new_pl_sd = OrderedDict()
        #     for key in state_dict['module'].keys():
        #         new_pl_sd[key[16:]]=state_dict['module'][key]
        #     model.load_state_dict(new_pl_sd, strict=full_strict)

        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
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
import os
from omegaconf import OmegaConf
import torch
import torchvision
from .scripts.evaluation.funcs import load_model_checkpoint, batch_ddim_sampling, get_latent_z
from .utils.utils import instantiate_from_config
from einops import repeat
import folder_paths
import comfy.model_management as mm
import comfy.utils
from contextlib import nullcontext

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError
    
script_directory = os.path.dirname(os.path.abspath(__file__))
class DynamiCrafterI2V:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "image": ("IMAGE",),
            "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "prompt": ("STRING", {"multiline": True, "default": "",}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "fs": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
            "dtype": (
                    [
                        'bf16',
                        'fp32',
                        'fp16',
                    ], {
                        "default": 'auto'
                    }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "DynamiCrafter"

    def process(self, image, dtype, ckpt_name, prompt, cfg, steps, eta, seed, fs):
        device = mm.get_torch_device()
        mm.unload_all_models()

        torch.manual_seed(seed)
        custom_config = {
            'dtype': dtype,
            'ckpt_name': ckpt_name,
        }
        dtype = convert_dtype(dtype)
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.current_config = custom_config
            model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            base_name, _ = os.path.splitext(ckpt_name)
            config_file=os.path.join(script_directory, "configs", f"{base_name}.yaml")
            config = OmegaConf.load(config_file)
            model_config = config.pop("model", OmegaConf.create())
            model_config['params']['unet_config']['params']['use_checkpoint']=False   
            self.model = instantiate_from_config(model_config)
            self.model = load_model_checkpoint(self.model, model_path)
            self.model.eval().to(dtype).to(device)

        channels = self.model.model.diffusion_model.out_channels
        frames = self.model.temporal_length
        B, H, W, C = image.shape

        noise_shape = [B, channels, frames, H // 8, W // 8]
        
        image = image * 2 - 1
        image = image.permute(0, 3, 1, 2).to(dtype).to(device)
        autocast_condition = (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            text_emb = self.model.get_learned_conditioning([prompt])

            z = get_latent_z(self.model, image.unsqueeze(2)) #bc,1,hw
            image
            img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)
            cond_images = self.model.embedder(image)
            img_emb = self.model.image_proj_model(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            fs = torch.tensor([fs], dtype=torch.long, device=self.model.device)
            cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
            ## inference
            batch_samples = batch_ddim_sampling(self.model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg)
            ## b,samples,c,t,h,w
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str=prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = 'empty_prompt'
        
        n_samples = batch_samples.shape[1]
        for idx, vid_tensor in enumerate(batch_samples):
            video = vid_tensor.detach().cpu()
            video = torch.clamp(video.float(), -1., 1.)
            video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
            grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
            grid = (grid + 1.0) / 2.0
            grid = grid.permute(0, 2, 3, 1)

        return (grid,)


NODE_CLASS_MAPPINGS = {
    "DynamiCrafterI2V": DynamiCrafterI2V,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamiCrafterI2V": "DynamiCrafterI2V",
}

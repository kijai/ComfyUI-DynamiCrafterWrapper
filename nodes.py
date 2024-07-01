import os
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from .scripts.evaluation.funcs import load_model_checkpoint, get_latent_z, get_latent_z_with_hidden_states
from .utils.utils import instantiate_from_config
from einops import repeat
import folder_paths
import comfy.model_management as mm
import comfy.utils
from contextlib import nullcontext
from .lvdm.models.samplers.ddim import DDIMSampler

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    is_accelerate_available = True
except:
    pass

def split_and_trim(input_string):
    # Split the string into an array using '|' as a separator
    array = input_string.split('|')
    
    # Trim white space from each element in the array
    trimmed_array = [element.strip() for element in array]
    
    return trimmed_array

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

class DownloadAndLoadDynamiCrafterModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [   'tooncrafter_512_interp-pruned-fp16.safetensors',
                        'dynamicrafter_512_fp16_pruned.safetensors',
                        'dynamicrafter_512_interp_fp16_pruned.safetensors',
                        'dynamicrafter_1024_fp16_pruned.safetensors',
                        'dynamicrafter-CIL-512-no-watermark-pruned-fp16.safetensors',
                    ],
                    {
                    "default": 'tooncrafter_512_interp-pruned-fp16.safetensors'
                    }),
            "dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto',
                    ], {
                        "default": 'auto'
                    }),
            "fp8_unet": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("DCMODEL",)
    RETURN_NAMES = ("DynCraft_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DynamiCrafterWrapper"

    def loadmodel(self, dtype, model, fp8_unet=False):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        custom_config = {
            'dtype': dtype,
            'ckpt_name': model,
            'fp8_unet': fp8_unet
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.current_config = custom_config
            download_path = os.path.join(folder_paths.models_dir, "checkpoints", "dynamicrafter")
            model_path = os.path.join(download_path, model)
            

            if not os.path.exists(model_path):
                print(f"Downloading model to: {model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="Kijai/DynamiCrafter_pruned", 
                                  allow_patterns=[f"*{model}*"],
                                  local_dir=download_path, 
                                  local_dir_use_symlinks=False)

            ckpt_base_name = os.path.basename(model_path)
            print(f"Loading model from: {model_path}")

            base_name, _ = os.path.splitext(ckpt_base_name)
            if 'toon' in base_name and '512' in base_name:
                config_file=os.path.join(script_directory, "configs", "tooncrafter_512_interp.yaml")
            elif 'interp' in base_name and '512' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_512_interp_v1.yaml")
            elif '1024' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_1024_v1.yaml")
            elif '512' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_512_v1.yaml")
            elif '256' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_256_v1.yaml")
            else:
                print(f"No matching config for model: {model}")
            config = OmegaConf.load(config_file)

            model_config = config.pop("model", OmegaConf.create())
            model_config['params']['unet_config']['params']['use_checkpoint']=False

            if dtype == "auto":
                try:
                    if mm.should_use_fp16():
                        precision = (convert_dtype('fp16'))
                    elif mm.should_use_bf16():
                        precision = (convert_dtype('bf16'))
                    else:
                        precision = (convert_dtype('fp32'))
                except:
                    raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
            else:
                precision = (convert_dtype(dtype))

            with (init_empty_weights() if is_accelerate_available else nullcontext()):
                self.model = instantiate_from_config(model_config)
            self.model = load_model_checkpoint(self.model, model_path, precision, device)
            self.model.to(precision).to(device).eval()
            
            if fp8_unet:
                self.model.model.diffusion_model = self.model.model.diffusion_model.to(torch.float8_e4m3fn)
            print(f"Model using dtype: {self.model.dtype}")

            dcmodel = {
                'model': self.model,
                'model_name': model,
            }
        return (dcmodel,)
    
class DownloadAndLoadCLIPModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [   'stable-diffusion-2-1-clip-fp16.safetensors',
                        'stable-diffusion-2-1-clip.safetensors',
                    ],
                    {
                    "default": 'stable-diffusion-2-1-clip-fp16.safetensors'
                    }),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "loadmodel"
    CATEGORY = "DynamiCrafterWrapper"

    def loadmodel(self, model):
        import shutil
        mm.soft_empty_cache()
        download_path = os.path.join(folder_paths.models_dir, "temp")
        model_path = os.path.join(folder_paths.models_dir, "clip", model)
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            filename = "model.fp16.safetensors" if "fp16" in model else "model.safetensors"
            subfolder = "text_encoder"
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1", 
                                subfolder = subfolder,
                                filename = filename,
                                local_dir=download_path, 
                                local_dir_use_symlinks=False)
            source_file_path = os.path.join(download_path, subfolder, filename)
            destination_file_path = model_path
            shutil.move(source_file_path, destination_file_path)

        clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        clip = comfy.sd.load_clip(ckpt_paths = [model_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type)

        print(f"Loading model from: {model_path}")

           
        return (clip,)

class DownloadAndLoadCLIPVisionModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [   'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors',
                        'CLIP-ViT-H-fp16.safetensors',
                    ],
                    {
                    "default": 'CLIP-ViT-H-fp16.safetensors'
                    }),
            },
        }

    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("clip_vision",)
    FUNCTION = "loadmodel"
    CATEGORY = "DynamiCrafterWrapper"

    def loadmodel(self, model):
        import shutil
        mm.soft_empty_cache()
        download_path = os.path.join(folder_paths.models_dir, "temp")
        model_path = os.path.join(folder_paths.models_dir, "clip_vision", model)
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import hf_hub_download
            if "fp16" in model:                
                hf_hub_download(repo_id="Kijai/CLIPVisionModelWithProjection_fp16", 
                                    filename = "CLIP-ViT-H-fp16.safetensors",
                                    local_dir = os.path.join(folder_paths.models_dir, "clip_vision"), 
                                    local_dir_use_symlinks=False)
            else:
                filename = "open_clip_pytorch_model.safetensors"
                hf_hub_download(repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
                                    filename = filename,
                                    local_dir=download_path, 
                                    local_dir_use_symlinks=False)
                source_file_path = os.path.join(download_path, filename)
                destination_file_path = model_path
                shutil.move(source_file_path, destination_file_path)

        clip_vision = comfy.clip_vision.load(model_path)

        print(f"Loading model from: {model_path}")
           
        return (clip_vision,)
    
class DynamiCrafterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto',
                    ], {
                        "default": 'auto'
                    }),
            "fp8_unet": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("DCMODEL",)
    RETURN_NAMES = ("DynCraft_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DynamiCrafterWrapper"

    def loadmodel(self, dtype, ckpt_name, fp8_unet=False):
        mm.soft_empty_cache()
        custom_config = {
            'dtype': dtype,
            'ckpt_name': ckpt_name,
            'fp8_unet': fp8_unet
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.current_config = custom_config
            model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            ckpt_base_name = os.path.basename(ckpt_name)
            base_name, _ = os.path.splitext(ckpt_base_name)
            if 'toon' in base_name and '512' in base_name:
                config_file=os.path.join(script_directory, "configs", "tooncrafter_512_interp.yaml")
            elif 'interp' in base_name and '512' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_512_interp_v1.yaml")
            elif '1024' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_1024_v1.yaml")
            elif '512' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_512_v1.yaml")
            elif '256' in base_name:
                config_file=os.path.join(script_directory, "configs", "dynamicrafter_256_v1.yaml")
            else:
                print(f"No matching config for model: {ckpt_name}")
            config = OmegaConf.load(config_file)

            model_config = config.pop("model", OmegaConf.create())
            model_config['params']['unet_config']['params']['use_checkpoint']=False
            self.model = instantiate_from_config(model_config)
            self.model = load_model_checkpoint(self.model, model_path)
            self.model.eval()
            if dtype == "auto":
                try:
                    if mm.should_use_fp16():
                        self.model.to(convert_dtype('fp16'))
                    elif mm.should_use_bf16():
                        self.model.to(convert_dtype('bf16'))
                    else:
                        self.model.to(convert_dtype('fp32'))
                except:
                    raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
            else:
                self.model.to(convert_dtype(dtype))
            if fp8_unet:
                self.model.model.diffusion_model = self.model.model.diffusion_model.to(torch.float8_e4m3fn)
            print(f"Model using dtype: {self.model.dtype}")
            dcmodel = {
                'model': self.model,
                'model_name': ckpt_name,
            }
        return (dcmodel,)
    
class DynamiCrafterI2V:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "clip_vision": ("CLIP_VISION", ),
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "image": ("IMAGE",),
            "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "frames": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "fs": ("INT", {"default": 10, "min": 2, "max": 100, "step": 1}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "vae_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask": ("MASK",),
                "frame_window_size": ("INT", {"default": 16, "min": 1, "max": 200, "step": 1}),
                "frame_window_stride": ("INT", {"default": 4, "min": 1, "max": 200, "step": 1}),
                "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "init_noise": ("DCNOISE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("images", "last_image",)
    FUNCTION = "process"
    CATEGORY = "DynamiCrafterWrapper"

    def process(self, model, image, clip_vision, positive, negative, cfg, steps, eta, seed, fs, keep_model_loaded, 
                frames, vae_dtype, frame_window_size=16, frame_window_stride=4, mask=None, image2=None, augmentation_level=0, init_noise=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()

        self.model = model['model']

        torch.manual_seed(seed)
        dtype = self.model.dtype
        if vae_dtype == "auto":
            try:
                if mm.should_use_bf16():
                    self.model.first_stage_model.to(convert_dtype('bf16'))
                else:
                    self.model.first_stage_model.to(convert_dtype('fp32'))
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
        else:
            self.model.first_stage_model.to(convert_dtype(vae_dtype))
        print(f"VAE using dtype: {self.model.first_stage_model.dtype}")

        
        self.model.to(device)
        autocast_condition = (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            image = image.permute(0, 3, 1, 2).to(dtype).to(device)
            if augmentation_level > 0:
                image += torch.randn_like(image) * augmentation_level

            B, C, H, W = image.shape
            orig_H, orig_W = H, W
            if W % 64 != 0:
                W = W - (W % 64)
            if H % 64 != 0:
                H = H - (H % 64)
            if orig_H % 64 != 0 or orig_W % 64 != 0:
                image = F.interpolate(image, size=(H, W), mode="bicubic")
           
            B, C, H, W = image.shape
            noise_shape = [B, self.model.model.diffusion_model.out_channels, frames, H // 8, W // 8]

            self.model.first_stage_model.to(device)
            encode_pixels = image.unsqueeze(2) * 2 - 1
            z = get_latent_z(self.model, encode_pixels) #bc,1,hw

            if image2 is not None:
                image2 = image2 * 2 - 1
                image2 = image2.permute(0, 3, 1, 2).to(dtype).to(device)

                if augmentation_level > 0:
                    image2 += torch.randn_like(image2) * augmentation_level

                if image2.shape != image.shape:
                    image2 = F.interpolate(image, size=(H, W), mode="bicubic")

                encode_pixels = image2.unsqueeze(2) * 2 - 1
                z2 = get_latent_z(self.model, encode_pixels) #bc,1,hw
                img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)
                img_tensor_repeat = torch.zeros_like(img_tensor_repeat)
                img_tensor_repeat[:,:,:1,:,:] = z
                img_tensor_repeat[:,:,-1:,:,:] = z2
            else:
                img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

            self.model.first_stage_model.to(offload_device)
     
            self.model.image_proj_model.to(device)
            text_emb = positive[0][0].to(device)

            cond_images = clip_vision.encode_image(image.permute(0, 2, 3, 1))['last_hidden_state'].to(device)

            img_emb = self.model.image_proj_model(cond_images)

            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            del cond_images, img_emb, text_emb, encode_pixels

            fs = torch.tensor([fs], dtype=torch.long, device=self.model.device)
            cond = {"c_crossattn": [imtext_cond], "c_concat": [img_tensor_repeat]}

            if noise_shape[-1] == 32:
                timestep_spacing = "uniform"
                guidance_rescale = 0.0
            else:
                timestep_spacing = "uniform_trailing"
                guidance_rescale = 0.7

            ## construct unconditional guidance
            if cfg != 1.0: 
                uc_emb = negative[0][0].to(device)
                ## process image embedding token
                if hasattr(self.model, 'embedder'):
                    uc_img = torch.rand(noise_shape[0],3,224,224).to(self.model.device)
                    ## img: b c h w >> b l c
                    uc_img = clip_vision.encode_image(uc_img.permute(0, 2, 3, 1))['last_hidden_state'].to(self.model.device)
                    uc_img = self.model.image_proj_model(uc_img)
                    uc_emb = torch.cat([uc_emb, uc_img], dim=1)
                if isinstance(cond, dict):
                    uc = {key:cond[key] for key in cond.keys()}
                    uc.update({'c_crossattn': [uc_emb]})
                else:
                    uc = uc_emb
            else:
                uc = None

            self.model.image_proj_model.to(offload_device)

            if mask is not None:     
                mask = mask.to(dtype).to(device)
                mask = F.interpolate(mask.unsqueeze(0), size=(H // 8, W // 8), mode="nearest").squeeze(0)
                mask = (1 - mask)
                mask = mask.unsqueeze(1)
                B, C, H, W = mask.shape
                if B < frames:
                    mask = mask.unsqueeze(2)
                    mask = mask.expand(-1, -1, frames, -1, -1)
                else:
                    mask = mask.unsqueeze(0)
                    mask = mask.permute(0, 2, 1, 3, 4) 
                mask = torch.where(mask < 1.0, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype))

            if init_noise is not None:
                if init_noise['analytic_init']:
                    eps=torch.randn_like(init_noise['mu_p'])
                    sigma_p = init_noise['sigma_p']
                    init = (init_noise['mu_p'] + sigma_p*eps).to(dtype).to(device)
                    if noise_shape[2] % init.shape[2] == 0:
                        init = init.repeat(1, 1, noise_shape[2] // init.shape[2], 1, 1)
                    else:
                        raise ValueError("The target dimension size is not an integral multiple of the original dimension size.")
                else:
                    init = None
                timestep_spacing = "uniform_trailing"
                guidance_rescale = 0.0
                ddpm_from = init_noise['M']
                
               
            else:
                init = None
                ddpm_from = 1000

            #inference
            ddim_sampler = DDIMSampler(self.model)
            samples, _ = ddim_sampler.sample(
                                            S=steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=True,
                                            unconditional_guidance_scale=cfg,
                                            unconditional_conditioning=uc,
                                            eta=eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=None,
                                            x_T=init,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            clean_cond=True,
                                            mask=mask,
                                            x0=img_tensor_repeat.clone() if mask is not None else None,
                                            frame_window_size = frame_window_size,
                                            frame_window_stride = frame_window_stride,
                                            noise_multiplier=1.0,
                                            ddpm_from=ddpm_from
                                            )
            
            assert not torch.isnan(samples).any().item(), "Resulting tensor containts NaNs. I'm unsure why this happens, changing step count and/or image dimensions might help."
            
            ## reconstruct from latent to pixel space
            self.model.first_stage_model.to(device)
            self.model.en_and_decode_n_samples_a_time = 1
            decoded_images = self.model.decode_first_stage(samples) #b c t h w
            self.model.first_stage_model.to(offload_device)
        
            video = decoded_images.detach().cpu()
            video = torch.clamp(video.float(), -1., 1.)
            video = (video + 1.0) / 2.0
            video = video.squeeze(0).permute(1, 2, 3, 0)
            del decoded_images, samples

            if not keep_model_loaded:
                self.model.to(offload_device)
                mm.soft_empty_cache()
            # Ensure the final dimensions are divisible by 2
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2

            if video.shape[1] != final_H or video.shape[2] != final_W:
                video = F.interpolate(video.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bicubic").permute(0, 2, 3, 1)
            last_image = video[-1].unsqueeze(0)
            return (video, last_image)
        
class DynamiCrafterLoadInitNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "M": ("INT", {"default": 1000, "min": 1, "max": 1000, "step": 1}),
            "analytic_init": ("BOOLEAN", {"default": True}),
        },
        }

    RETURN_TYPES = ("DCNOISE", "INT", "INT",)
    RETURN_NAMES = ("init_noise", "width", "height",)
    FUNCTION = "load"
    CATEGORY = "DynamiCrafterWrapper"

    def load(self, model, M, analytic_init):
        device = mm.get_torch_device()
        
        model_name = model['model_name']
        if '512' in model_name:
            analytic_noise = "initial_noise_512.safetensors"
        elif '1024' in model_name:
            analytic_noise = "initial_noise_1024.safetensors"
        else:
            print("Can't find matching init_noise for model: ", model_name)
        model_path = os.path.join(script_directory, 'init_noises', analytic_noise)

        # Analytic-Init:load initial noise 
        dic = comfy.utils.load_torch_file(model_path)
        expectation_X_0=dic["Expectation_X0"].to(device)
        tr_Cov_d=dic["Tr_Cov_d"].to(device)
        
        sqrt_alpha_t=model['model'].get_sqrt_alpha_t_bar(expectation_X_0,torch.tensor([M-1]).to(device))
        mu_p=sqrt_alpha_t*expectation_X_0
        alpha_t=sqrt_alpha_t**2
        sigma_p=torch.sqrt(1-alpha_t + alpha_t*tr_Cov_d)

        init_noise = {
            "sigma_p": sigma_p,
            "mu_p": mu_p,
            "M": M,
            "analytic_init": analytic_init
            }
        width = mu_p.shape[4] * 8
        height = mu_p.shape[3] * 8
       
        return (init_noise, width, height)

class ToonCrafterInterpolation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "clip_vision": ("CLIP_VISION", ),
            "positive": ("CONDITIONING", ),
            "negative": ("CONDITIONING", ),
            "images": ("IMAGE",),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 200.0, "step": 0.01}),
            "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "frames": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "fs": ("INT", {"default": 10, "min": 2, "max": 100, "step": 1}),
            "vae_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            },
            "optional": {
                "image_embed_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "optional_latents": ("LATENT",),
                "latent_noise_multiplier": ("FLOAT", {"default": 1.0, "min": 0, "max": 100, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "DynamiCrafterWrapper"

    def process(self, model, clip_vision, images, positive, negative, cfg, steps, eta, seed, fs, frames, vae_dtype, image_embed_ratio=1.0, augmentation_level=0, optional_latents=None, latent_noise_multiplier=1.0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()

        torch.manual_seed(seed)

        self.model = model['model']

        dtype = self.model.dtype
        if vae_dtype == "auto":
            try:
                if mm.should_use_bf16():
                    self.model.first_stage_model.to(convert_dtype('bf16'))
                else:
                    self.model.first_stage_model.to(convert_dtype('fp32'))
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
        else:
            self.model.first_stage_model.to(convert_dtype(vae_dtype))
        print(f"VAE using dtype: {self.model.first_stage_model.dtype}")

        images = images.permute(0, 3, 1, 2).to(dtype).to(device)

        B, C, H, W = images.shape
        orig_H, orig_W = H, W
        if W % 64 != 0:
            W = W - (W % 64)
        if H % 64 != 0:
            H = H - (H % 64)
        if orig_H % 64 != 0 or orig_W % 64 != 0:
            images = F.interpolate(images, size=(H, W), mode="bicubic")

        self.model.to(device)

        out = []
        hidden_states = []
        pbar = comfy.utils.ProgressBar(len(images) - 1)
        autocast_condition = (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for i in range(len(images) - 1):
                videos, videos2 = None, None
                mm.soft_empty_cache()
                image = images[i].unsqueeze(0)
                image2 = images[i+1].unsqueeze(0)
                
                B, C, H, W = image.shape
                noise_shape = [B, self.model.model.diffusion_model.out_channels, frames, H // 8, W // 8]

                self.model.first_stage_model.to(device)

                if augmentation_level > 0:
                    image += torch.randn_like(image) * augmentation_level
                    image2 += torch.randn_like(image) * augmentation_level

                encode_pixels = image.unsqueeze(2) * 2 - 1
                videos = encode_pixels # bc1hw
                videos = repeat(videos, 'b c t h w -> b c (repeat t) h w', repeat=frames//2)
                encode_pixels = image2.unsqueeze(2) * 2 - 1
                videos2 = encode_pixels # bc1hw
                videos2 = repeat(videos2, 'b c t h w -> b c (repeat t) h w', repeat=frames//2)
                videos = torch.cat([videos, videos2], dim=2)                              

                try:
                    z, hs = get_latent_z_with_hidden_states(self.model, videos)
                    hs = [t.to("cpu") for t in hs]
                    hidden_states.append(hs)
                except:
                    z = get_latent_z(self.model, videos)
                    hidden_states = None

                img_tensor_repeat = torch.zeros_like(z)
                img_tensor_repeat[:,:,:1,:,:] = z[:,:,:1,:,:]
                img_tensor_repeat[:,:,-1:,:,:] = z[:,:,-1:,:,:]

                self.model.first_stage_model.to(offload_device)

                text_emb = positive[0][0].to(device)
                
                cond_images = clip_vision.encode_image(image.permute(0, 2, 3, 1))["last_hidden_state"].to(device)
                cond_images2 = clip_vision.encode_image(image2.permute(0, 2, 3, 1))["last_hidden_state"].to(device)

                self.model.image_proj_model.to(device)

                img_emb = self.model.image_proj_model(cond_images)
                img_emb2 = self.model.image_proj_model(cond_images2)
                img_embeds = img_emb * image_embed_ratio + img_emb2 * (1.0 - image_embed_ratio)

                imtext_cond = torch.cat([text_emb, img_embeds], dim=1)
                del cond_images, img_emb, img_emb2, text_emb

                if comfy.model_management.is_device_mps(device):
                    fs = torch.tensor([fs], dtype=torch.float32, device=self.model.device)
                else:
                    fs = torch.tensor([fs], dtype=torch.float64, device=self.model.device)
                cond = {"c_crossattn": [imtext_cond], "c_concat": [img_tensor_repeat]}

                if noise_shape[-1] == 32:
                    timestep_spacing = "uniform"
                    guidance_rescale = 0.0
                else:
                    timestep_spacing = "uniform_trailing"
                    guidance_rescale = 0.7

                ## construct unconditional guidance
                if cfg != 1.0: 
                    uc_emb = negative[0][0].to(device)
                    ## process image embedding token
                    if hasattr(self.model, 'embedder'):
                        uc_img = torch.rand(noise_shape[0],3,224,224).to(self.model.device)
                        ## img: b c h w >> b l c
                        uc_img = clip_vision.encode_image(uc_img.permute(0, 2, 3, 1))['last_hidden_state'].to(self.model.device)
                        uc_img = self.model.image_proj_model(uc_img)
                        uc_emb = torch.cat([uc_emb, uc_img], dim=1)
                    if isinstance(cond, dict):
                        uc = {key:cond[key] for key in cond.keys()}
                        uc.update({'c_crossattn': [uc_emb]})
                    else:
                        uc = uc_emb
                else:
                    uc = None

                self.model.image_proj_model.to(offload_device)

                #inference
                if optional_latents is not None:
                    samples_in = optional_latents['samples'].clone().to(device)
                    samples_in = samples_in * 0.18215
                    samples_in = samples_in.unsqueeze(0).permute(0, 2, 1, 3, 4)
                    noise = torch.randn(noise_shape, device=device)
                    samples_in[:, :, 0, :, :] = noise[:, :, 0, :, :]
                    samples_in[:, :, -1, :, :] = noise[:, :, -1, :, :]
                    samples_in = samples_in.to(dtype).to(device)
                else:
                    samples_in = None

                self.model.model.diffusion_model.to(device)
                ddim_sampler = DDIMSampler(self.model)
                samples, _ = ddim_sampler.sample(S=steps,
                                                conditioning=cond,
                                                batch_size=noise_shape[0],
                                                shape=noise_shape[1:],
                                                verbose=True,
                                                unconditional_guidance_scale=cfg,
                                                unconditional_conditioning=uc,
                                                eta=eta,
                                                temporal_length=noise_shape[2],
                                                conditional_guidance_scale_temporal=None,
                                                x_T=samples_in,
                                                fs=fs,
                                                timestep_spacing=timestep_spacing,
                                                guidance_rescale=guidance_rescale,
                                                clean_cond=True,
                                                mask=None,
                                                x0=None,
                                                frame_window_size = 16,
                                                frame_window_stride = 4,
                                                noise_multiplier = latent_noise_multiplier
                                                )
                print(f"Sampled {i+1} out of {(len(images) - 1)}")
                assert not torch.isnan(samples).any().item(), "Resulting tensor containts NaNs. I'm unsure why this happens, changing step count and/or image dimensions might help."
                samples = samples.squeeze(0).permute(1, 0, 2, 3).cpu().to(self.model.first_stage_model.dtype)
                out.append(samples)
                pbar.update(1)

            self.model.to(offload_device)
            mm.soft_empty_cache()

            samples = torch.cat(out, dim=0)
            samples = samples / 0.18215

            latent = {
                "samples": samples,
                "hidden_states": hidden_states,
                }

            return (latent,)

class ToonCrafterDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "latent": ("LATENT",),
            "vae_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            },
              "optional": {
                "prune_last_frame": ("BOOLEAN", {"default": False}),
              }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "DynamiCrafterWrapper"

    def process(self, model, latent, vae_dtype, prune_last_frame=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        self.model = model['model']
        samples = latent["samples"]
        num_samples = samples.shape[0]
        samples = samples * 0.18215
        self.model.first_stage_model.to(device)
        #samples = samples.to(model.first_stage_model.device)

        hs = latent["hidden_states"]

        self.model.en_and_decode_n_samples_a_time = 16

        if vae_dtype == "auto":
            try:
                if mm.should_use_bf16():
                    self.model.first_stage_model.to(convert_dtype('bf16'))
                else:
                    self.model.first_stage_model.to(convert_dtype('fp32'))
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
        else:
            self.model.first_stage_model.to(convert_dtype(vae_dtype))
        print(f"VAE using dtype: {self.model.first_stage_model.dtype}")
        out = []
        iteration_counter = 0
        pbar = comfy.utils.ProgressBar(num_samples // 16)
        autocast_condition = (self.model.first_stage_model.dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
        for i in range(0, num_samples, 16):
            batch_start = i
            batch_end = min(i + 16, num_samples)  # Ensure we don't go beyond the tensor's size
            batch_samples = samples[batch_start:batch_end].to(self.model.first_stage_model.device)
            with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=self.model.first_stage_model.dtype) if autocast_condition else nullcontext():
                if mm.XFORMERS_IS_AVAILABLE:
                    print(f"Decoding frames {iteration_counter * 16} - {16 + iteration_counter * 16} out of {num_samples} using xformers")
                    if hs is not None:
                        hs_ = hs[iteration_counter]
                        hs_ = [t.to(self.model.first_stage_model.device) for t in hs_]
                        additional_decode_kwargs = {'ref_context': hs_}
                        decoded_images = self.model.decode_first_stage(batch_samples, **additional_decode_kwargs) #b c t h w
                    else:
                        decoded_images = self.model.decode_first_stage(batch_samples) #b c t h w     
                else:
                    raise Exception("XFormers not available, it is required for ToonCrafter decoder. Alternatively you can use a standard VAE Decode -node instead, but this has a negative effect on the image quality though.")
                
                video = decoded_images.detach().cpu()
                video = torch.clamp(video.float(), -1., 1.)
                video = (video + 1.0) / 2.0
                video = video.squeeze(0).permute(0, 2, 3, 1)
                iteration_counter += 1
                pbar.update(1)
                out.append(video)
                del decoded_images
                mm.soft_empty_cache()
        self.model.first_stage_model.to(offload_device)
        video_out = torch.cat(out, dim=0)
        if prune_last_frame:
            video_out = video_out[torch.arange(video_out.shape[0]) % 16!= 0]

        return (video_out,)
                
class DynamiCrafterBatchInterpolation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("DCMODEL",),
            "clip_vision": ("CLIP_VISION",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "images": ("IMAGE",),
            "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "frames": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "fs": ("INT", {"default": 10, "min": 2, "max": 100, "step": 1}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "vae_dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            "cut_near_keyframes": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("images", "last_image",)
    FUNCTION = "process"
    CATEGORY = "DynamiCrafterWrapper"

    def process(self, model, images, clip_vision, positive, negative, cfg, steps, eta, seed, fs, keep_model_loaded,
                frames, vae_dtype, cut_near_keyframes):
        assert images.shape[0] > 1, "DynamiCrafterBatchInterpolation needs at least 2 images"
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()

        torch.manual_seed(seed)
        dtype = model.dtype
        self.model = model['model']

        if vae_dtype == "auto":
            try:
                if mm.should_use_bf16():
                    self.model.first_stage_model.to(convert_dtype('bf16'))
                else:
                    self.model.first_stage_model.to(convert_dtype('fp32'))
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtype manually.")
        else:
            self.model.first_stage_model.to(convert_dtype(vae_dtype))
        print(f"VAE using dtype: {self.model.first_stage_model.dtype}")
      
        self.model.to(device)
        images = images * 2 - 1
        images = images.permute(0, 3, 1, 2).to(dtype).to(device)
        B, C, H, W = images.shape
        orig_H, orig_W = H, W
        if W % 64 != 0:
            W = W - (W % 64)
        if H % 64 != 0:
            H = H - (H % 64)
        if orig_H % 64 != 0 or orig_W % 64 != 0:
            images = F.interpolate(images, size=(H, W), mode="bicubic")        
		
        out = []
        autocast_condition = (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for i in range(len(images) - 1):

                image = images[i].unsqueeze(0)
                image2 = images[i+1].unsqueeze(0)
                B, C, H, W = image.shape
                noise_shape = [B, self.model.model.diffusion_model.out_channels, frames, H // 8, W // 8]

                self.model.first_stage_model.to(device)

                z = get_latent_z(self.model, image.unsqueeze(2)) #bc,1,hw
                z2 = get_latent_z(self.model, image2.unsqueeze(2)) #bc,1,hw
                img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)
                img_tensor_repeat = torch.zeros_like(img_tensor_repeat)
                img_tensor_repeat[:,:,:1,:,:] = z
                img_tensor_repeat[:,:,-1:,:,:] = z2
         
                self.model.first_stage_model.to('cpu')

                self.model.embedder.to(device)
                self.model.image_proj_model.to(device)
                text_emb = positive[0][0].to(device)
                cond_images = clip_vision.encode_image(image.permute(0, 2, 3, 1))['last_hidden_state'].to(device)

                img_emb = self.model.image_proj_model(cond_images)
                imtext_cond = torch.cat([text_emb, img_emb], dim=1)

                fs = torch.tensor([fs], dtype=torch.long, device=self.model.device)
                cond = {"c_crossattn": [imtext_cond], "c_concat": [img_tensor_repeat]}

                if noise_shape[-1] == 32:
                    timestep_spacing = "uniform"
                    guidance_rescale = 0.0
                else:
                    timestep_spacing = "uniform_trailing"
                    guidance_rescale = 0.7

                ## construct unconditional guidance
                if cfg != 1.0:
                    uc_emb = negative[0][0].to(device)
                    ## process image embedding token
                    if hasattr(self.model, 'embedder'):
                        uc_img = torch.rand(noise_shape[0], 3, 224, 224).to(self.model.device)
                        ## img: b c h w >> b l c
                        uc_img = clip_vision.encode_image(uc_img.permute(0, 2, 3, 1))['last_hidden_state'].to(
                            self.model.device)
                        uc_img = self.model.image_proj_model(uc_img)
                        uc_emb = torch.cat([uc_emb, uc_img], dim=1)
                    if isinstance(cond, dict):
                        uc = {key:cond[key] for key in cond.keys()}
                        uc.update({'c_crossattn': [uc_emb]})
                    else:
                        uc = uc_emb
                else:
                    uc = None
                self.model.embedder.to('cpu')
                self.model.image_proj_model.to('cpu')

                #inference
                ddim_sampler = DDIMSampler(self.model)
                samples, _ = ddim_sampler.sample(S=steps,
                                                conditioning=cond,
                                                batch_size=noise_shape[0],
                                                shape=noise_shape[1:],
                                                verbose=True,
                                                unconditional_guidance_scale=cfg,
                                                unconditional_conditioning=uc,
                                                eta=eta,
                                                temporal_length=noise_shape[2],
                                                conditional_guidance_scale_temporal=None,
                                                x_T=None,
                                                fs=fs,
                                                timestep_spacing=timestep_spacing,
                                                guidance_rescale=guidance_rescale,
                                                clean_cond=True
                                                )
                
                assert not torch.isnan(samples).any().item(), "Resulting tensor containts NaNs. I'm unsure why this happens, changing step count and/or image dimensions might help."
                ## reconstruct from latent to pixel space
                self.model.first_stage_model.to(device)
                decoded_images = self.model.decode_first_stage(samples) #b c t h w
                self.model.first_stage_model.to('cpu')
            
                video = decoded_images.detach().cpu()
                video = torch.clamp(video.float(), -1., 1.)
                video = (video + 1.0) / 2.0
                video = video.squeeze(0).permute(1, 2, 3, 0)
                print(f"Sampled {i+1} / {len(images) - 1}")
                out.append(video)

            if not keep_model_loaded:
                self.model.to('cpu')
                mm.soft_empty_cache()
            out_video = torch.cat(out, dim=0)

            # Ensure the final dimensions are divisible by 2
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2

            if out_video.shape[1] != final_H or out_video.shape[2] != final_W:
                out_video = F.interpolate(out_video.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bicubic").permute(0, 2, 3, 1)

            # should we trim middle keyframes?
            if cut_near_keyframes > 0:
                already_deleted = 0
                for i in range(len(images) - 2):
                    old_size = out_video.shape[0]
                    keyframe_index = (i + 1) * frames - already_deleted
                    start_index = keyframe_index - (cut_near_keyframes // 2)
                    end_index = start_index + cut_near_keyframes
                    out_video = torch.cat([out_video[:start_index], out_video[end_index:]], dim=0)
                    already_deleted += old_size - out_video.shape[0]

            # should we trim middle keyframes?
            if cut_near_keyframes > 0:
                already_deleted = 0
                for i in range(len(images) - 2):
                    old_size = out_video.shape[0]
                    keyframe_index = (i + 1) * frames - already_deleted
                    start_index = keyframe_index - (cut_near_keyframes // 2)
                    end_index = start_index + cut_near_keyframes
                    out_video = torch.cat([out_video[:start_index], out_video[end_index:]], dim=0)
                    already_deleted += old_size - out_video.shape[0]

            last_image = out_video[-1].unsqueeze(0)
            return (out_video, last_image)

NODE_CLASS_MAPPINGS = {
    "DynamiCrafterI2V": DynamiCrafterI2V,
    "DynamiCrafterModelLoader": DynamiCrafterModelLoader,
    "DynamiCrafterBatchInterpolation": DynamiCrafterBatchInterpolation,
    "ToonCrafterInterpolation": ToonCrafterInterpolation,
    "ToonCrafterDecode": ToonCrafterDecode,
    "DownloadAndLoadDynamiCrafterModel": DownloadAndLoadDynamiCrafterModel,
    "DownloadAndLoadCLIPModel": DownloadAndLoadCLIPModel,
    "DownloadAndLoadCLIPVisionModel": DownloadAndLoadCLIPVisionModel,
    "DynamiCrafterLoadInitNoise": DynamiCrafterLoadInitNoise

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamiCrafterI2V": "DynamiCrafterI2V",
    "DynamiCrafterModelLoader": "DynamiCrafterModelLoader",
    "DynamiCrafterBatchInterpolation": "DynamiCrafterBatchInterpolation",
    "ToonCrafterInterpolation": "ToonCrafterInterpolation",
    "ToonCrafterDecode": "ToonCrafterDecode",
    "DownloadAndLoadDynamiCrafterModel": "DownloadAndLoadDynamiCrafterModel",
    "DownloadAndLoadCLIPModel": "DownloadAndLoadCLIPModel",
    "DownloadAndLoadCLIPVisionModel": "DownloadAndLoadCLIPVisionModel",
    "DynamiCrafterLoadInitNoise": "DynamiCrafterLoadInitNoise"
}

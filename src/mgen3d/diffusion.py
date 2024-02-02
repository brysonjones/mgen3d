from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    logging,
    CLIPVisionModel,
    CLIPImageProcessor,
)
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as T
import time
import os
import yaml


class Diffusion(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(
            config["components"]["diffusion_key"], subfolder="vae"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            config["components"]["diffusion_key"], subfolder="unet"
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            config["components"]["diffusion_key"], subfolder="scheduler"
        )

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

    def get_num_train_timesteps(self):
        return self.scheduler.config.num_train_timesteps

    def produce_latents(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.config.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast("cuda"):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents


class CLIP(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.device = device

        # Create model
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config["components"]["diffusion_key"], subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            config["components"]["diffusion_key"], subfolder="text_encoder"
        ).to(self.device)
        self.image_encoder = CLIPVisionModel.from_pretrained(
            config["components"]["clip_key"]
        ).to(self.device)
        self.text_clip_encoder = CLIPVisionModel.from_pretrained(
            config["components"]["clip_key"]
        ).to(self.device)
        self.processor = CLIPImageProcessor.from_pretrained(
            config["components"]["clip_key"]
        )

        self.aug = T.Compose(
            [
                T.Resize((224, 224)),
                T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def img_clip_loss(self, rgb1, rgb2):
        image_z_1 = self.image_encoder(self.aug(rgb1))
        image_z_2 = self.image_encoder(self.aug(rgb2))
        image_z_1 = image_z_1 / image_z_1.norm(
            dim=-1, keepdim=True
        )  # normalize features
        image_z_2 = image_z_2 / image_z_2.norm(
            dim=-1, keepdim=True
        )  # normalize features

        loss = -(image_z_1 * image_z_2).sum(-1).mean()
        return loss

    def img_text_clip_loss(self, rgb, prompt):
        image_z_1 = self.image_encoder(self.aug(rgb))
        image_z_1 = image_z_1 / image_z_1.norm(
            dim=-1, keepdim=True
        )  # normalize features

        text = self.tokenizer(prompt).to(self.device)
        text_z = self.text_encoder(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)
        loss = -(image_z_1 * text_z).sum(-1).mean()
        return loss


class StableDiffusion(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.device = device
        self.diffusion = Diffusion(config, device)
        self.clip = CLIP(config, device)

        self.num_train_timesteps = self.diffusion.get_num_train_timesteps()
        self.num_inference_steps = 50
        self.min_step = int(
            self.num_train_timesteps * float(config["denoising"]["min_step"])
        )
        self.max_step = int(
            self.num_train_timesteps * float(config["denoising"]["max_step"])
        )

    def train_step(
        self,
        text_embeddings,
        pred_rgb,
        ref_rgb=None,
        noise=None,
        islarge=False,
        ref_text=None,
        guidance_scale=10,
    ):
        # interp to 512x512 to be fed into vae.
        loss = 0
        imgs = None

        pred_rgb_512 = F.interpolate(
            pred_rgb, (512, 512), mode="bilinear", align_corners=False
        )

        t_sample = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )

        # encode image into latents with vae
        latents = self.encode_imgs(pred_rgb_512)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t_sample)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = latent_model_input.detach().requires_grad_()

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if not islarge and (t_sample / self.num_train_timesteps) <= 0.4:
            self.scheduler.set_timesteps(self.num_train_timesteps)
            de_latents = self.scheduler.step(noise_pred, t_sample, latents_noisy)[
                "prev_sample"
            ]
            imgs = self.decode_latents(de_latents)
            loss = 10 * self.img_clip_loss(
                imgs, ref_rgb
            ) + 10 * self.img_text_clip_loss(imgs, ref_text)

        else:
            # w(t), sigma_t^2
            w = 1 - self.alphas[t]
            grad = w * (noise_pred - noise)
            imgs = None

            # clip grad for stable training?
            grad = torch.nan_to_num(grad)
            latents.backward(gradient=grad, retain_graph=True)
            loss = 0

        return loss, imgs  # dummy loss value

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        text_embeds = self.clip.get_text_embeds(
            prompts, negative_prompts
        )  # [2, 77, 768]
        latents = self.diffusion.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]
        image = self.diffusion.decode_latents(latents)  # [1, 3, 512, 512]
        image = image.squeeze()
        return image


def main():
    import sys
    import absl.flags as flags

    FLAGS = flags.FLAGS
    flags.DEFINE_string("config_path", None, "Path to the config file")
    flags.mark_flag_as_required("config_path")
    flags.FLAGS(sys.argv)

    with open(FLAGS.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    output_path = config["workspace"]["path"]
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd = StableDiffusion(config, device)
    prompt = config["generation"]["prompt"]
    guidance_scale = config["generation"]["guidance_scale"]

    imgs = sd.prompt_to_img(prompts=prompt, guidance_scale=guidance_scale)
    save_image(
        imgs,
        os.path.join(output_path, prompt.replace(" ", "_") + ".png"),
    )

if __name__ == "__main__":
    main()

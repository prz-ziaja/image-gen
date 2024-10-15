import torch
import torch.nn as nn

class imageDiffuser(nn.Module):
    """
    The class aims to produce noisy images for stable diffusion model.
    """
    def __init__(self, T:int):
        """
        T - max timestamp - maximum number steps during there is added noise.
        """
        nn.Module.__init__(self)
        self.T = T

        start = 1e-4
        end = 2e-2
        noise_schedule= torch.linspace(start, end, T)
        image_schedule = 1.0 - noise_schedule

        cumprod_image_sch = torch.cumprod(image_schedule, dim=0)
        self.image_coefs = nn.Parameter(torch.sqrt(cumprod_image_sch), requires_grad=False)
        self.noise_coefs = nn.Parameter(torch.sqrt(1-cumprod_image_sch), requires_grad=False)

    # def to(self, device):
    #     self.image_coefs = self.image_coefs.to(device)
    #     self.noise_coefs = self.noise_coefs.to(device)
    #     self.device = device
    #     return self
    def reverse(self, noisy_image, noise, t):
        image_coef= self.image_coefs[t, None, None]
        noise_coef = self.noise_coefs[t, None, None]
        image = (noisy_image -noise_coef * noise) /image_coef
        return image

    def __call__(self, image:torch.Tensor, t:list):
        """
        returns a noisy image.
        image - orginal image
        t - timestep - the bigger the more noise. Must be passed as iterable object of ints (eg one dimensional tensor)
        """
        noise = torch.randn_like(image, device=self.image_coefs.device)
        image_coef= self.image_coefs[t, None, None]
        noise_coef = self.noise_coefs[t, None, None]

        noisy_image = image_coef * image + noise_coef * noise
        return noisy_image, noise
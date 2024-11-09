import torch
import torch.nn as nn


class imageDiffuser(nn.Module):
    def __init__(self, T: int, t_start: float = 1e-4, t_end: float = 2e-2):
        """
        T - max timestamp - maximum number steps during there is added noise.
        """
        nn.Module.__init__(self)

        self.T = T

        self.noise_schedule = nn.Parameter(
            torch.linspace(t_start, t_end, T), requires_grad=False
        )
        image_schedule = 1.0 - self.noise_schedule

        # Forward diffusion params
        cumprod_image_sch = torch.cumprod(image_schedule, dim=0)
        self.image_coefs = nn.Parameter(
            torch.sqrt(cumprod_image_sch), requires_grad=False
        )
        self.noise_coefs = nn.Parameter(
            torch.sqrt(1 - cumprod_image_sch), requires_grad=False
        )

        # Reverse diffusion variables
        self.rev_image_coefs = nn.Parameter(
            torch.sqrt(1 / image_schedule), requires_grad=False
        )
        self.rev_pred_noise_coefs = nn.Parameter(
            (1 - image_schedule) / torch.sqrt(1 - self.image_coefs), requires_grad=False
        )

    @torch.no_grad()
    def reverse(self, noisy_image: torch.tensor, noise: torch.tensor, t: int):
        """
        Only for inference purpose - supports generating of a batch of images
        under condtion that all images are at the same timestep t and all are of the same shape.
        """
        pred_noise_coef = self.rev_pred_noise_coefs[t]
        image_coef = self.rev_image_coefs[t]
        denoised = image_coef * (noisy_image - pred_noise_coef * noise)
        if t == 0:
            return denoised
        else:
            B_t = self.noise_schedule[t - 1]  # Apply noise from the previos timestep
            new_noise = torch.randn_like(noisy_image)
            return denoised + torch.sqrt(B_t) * new_noise

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, t: torch.tensor):
        """
        returns a noisy image.
        image - batch of orginal images. Shape of tensor -> [Batch_size, ch, h, w]
        t - timestep - the bigger the more noise. Shape of tensor -> [Batch_size, 1]
        """
        noise = torch.randn_like(image, device=self.image_coefs.device)
        image_coef = self.image_coefs[t, None, None]
        noise_coef = self.noise_coefs[t, None, None]

        noisy_image = image_coef * image + noise_coef * noise
        return noisy_image, noise

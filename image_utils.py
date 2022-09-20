import torch
import torchvision.transforms.functional as F
from torch.nn import functional as TF
import numpy as np
from torchvision import transforms
import torchvision


def pgd_attack(model, images, labels, eps, alpha) :
    loss = torch.nn.CrossEntropyLoss()
        
    ori_images = images.data
    iters=40    
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

def fgsm_attack(model, images, labels, eps) :
    loss = torch.nn.CrossEntropyLoss()
        
    ori_images = images.data  
    images.requires_grad = True
    outputs = model(images)

    model.zero_grad()
    cost = loss(outputs, labels).cuda()
    cost.backward()

    adv_images = images + alpha*images.grad.sign()
    eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
    images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images


class PGD_DEC(torch.nn.Module):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, detector, eps=32/255,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.detector = detector
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach()
        labels = labels.clone().detach()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.detector(adv_images)

            # Calculate loss
            cost = self.ce_loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
class FGSM_DEC(torch.nn.Module):
    def __init__(self, detector, eps):
        super().__init__()
        self.eps = eps
        self.detector = detector
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, images, labels):
        images = images.clone().detach()
        labels = labels.clone().detach()
        images.requires_grad = True
        outputs = self.detector(images)
        cost = self.ce_loss(outputs, labels)
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        attack_images = images + self.eps* grad.sign()
        attack_images = torch.clamp(attack_images, min=0, max=1).detach()
        return attack_images

class PGD_ENC_DEC(torch.nn.Module):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, encoder, detector, eps=8/255,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.encoder = encoder
        self.detector = detector

    def forward(self, images, secret_input, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach()
        labels = labels.clone().detach()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.encoder(adv_images, secret_input)
            outputs = self.detector(outputs)

            # Calculate loss
            cost = TF.binary_cross_entropy(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
class FGSM_ENC_DEC(torch.nn.Module):
    def __init__(self, encoder, detector, eps=0.3):
        super().__init__()
        self.eps = eps
        self.encoder = encoder
        self.detector = detector

    def forward(self, images, secret_input, labels):
        images = images.clone().detach()
        labels = labels.clone().detach()
        images.requires_grad = True
        outputs = self.encoder(images,secret_input)
        outputs = self.detector(outputs)
        cost = TF.binary_cross_entropy(outputs, labels)
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        attack_images = images + self.eps* grad.sign()
        attack_images = torch.clamp(attack_images, min=0, max=1).detach()
        return attack_images
    
class Image_Operations():
    def __init__(self):
        super().__init__()

    def uni_noise_perturb(self,image, eps):
        self.image = image
        self.image = self.image.clone().detach()
        noise = torch.empty_like(self.image).uniform_(-eps, eps)
        perturbed_image = self.image + noise
        return perturbed_image
    
    def gaussian_noise_perturb(self, image, eps):
        self.image = image
        self.image = self.image.clone().detach()
        noise = torch.rand_like(self.image).cuda()*eps
        noise = torch.clamp(noise, min=-eps, max=eps)
        perturbed_image = self.image + noise
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1)
        return perturbed_image
    
    def rotate(self, image, degrees):
        self.image = image
        r"""
        degrees is a 2D list. e.g., degrees=[-20, 20]
        """
        angle = np.random.uniform(degrees[0], degrees[1])
        image = self.image.cpu()
        adv_image = torch.empty_like(image)
        to_pil = torchvision.transforms.ToPILImage()
        to_tensor = torchvision.transforms.ToTensor()
        resizer = torchvision.transforms.Resize((adv_image.size()[-2], adv_image.size()[-1]))
        for i in range(self.image.size()[0]):
            image = to_pil(self.image[i].cpu())
            perturbed_image = torchvision.transforms.functional.rotate(image, angle, fill=255, resample=2, expand=False)
            perturbed_image = resizer(perturbed_image)
            perturbed_image = to_tensor(perturbed_image)
            adv_image[i,:,:,:] = perturbed_image
        return adv_image.cuda()
    
    def center_crop(self, image, size_range):
        self.image = image
        r"""
        size_range is a 2D list, e.g., size_range=[250, 400]
        """
        crop_size = int(np.random.uniform(size_range[0], size_range[1]))
        image = self.image.cpu()
        adv_image = torch.empty_like(image)
        to_pil = torchvision.transforms.ToPILImage()
        to_tensor = torchvision.transforms.ToTensor()
        resizer = torchvision.transforms.Resize((adv_image.size()[-2], adv_image.size()[-1]))
        for i in range(self.image.size()[0]):
            image = to_pil(self.image[i].cpu())
            perturbed_image = torchvision.transforms.functional.center_crop(image, crop_size)
            perturbed_image = resizer(perturbed_image)
            perturbed_image = to_tensor(perturbed_image)
            adv_image[i,:,:,:] = perturbed_image
        return adv_image.cuda()

def random_crop(image, starts, size):
    start1,start2 = starts
    mask = torch.zeros_like(image)
    mask[:,:,start1:start1+size+1,start2:start2+size+1] = mask[:,:,start1:start1+size+1,start2:start2+size+1]+1
    new_image = torch.mul(image, mask)
    return new_image
        
        
        
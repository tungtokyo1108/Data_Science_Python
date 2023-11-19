#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:47:40 2023

@author: tungdang
"""

## Standard libraries
import os
import json
import math
import numpy as np
import random

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
sns.reset_orig()

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

##################################################################################################################

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "models_tutorial8"

device = torch.device("cpu")

# Transformations applied on each image => make them a tensor and normalize between -1 and 1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                               ])

# Loading the training dataset. We need to split it into a training and validation part
train_set = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)

# Loading the test set
test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
# Note that for actually training a model, we will use different data loaders
# with a lower batch size.
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
test_loader  = data.DataLoader(test_set,  batch_size=256, shuffle=False, drop_last=False, num_workers=4)

##################################################################################################################

class Swish(nn.Module):
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class CNNModel(nn.Module):
    
    def __init__(self, hidden_features=32, out_dim=1, **kwargs):
        
        super().__init__()
        
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features 
        c_hid3 = hidden_features**2
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid3*4, c_hid3),
            Swish(),
            nn.Linear(c_hid3, out_dim)
            )
    
    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim = -1)
        return x
    
class Sampler:
    
    def __init__(self, model, img_shape, sample_size, max_len=8192):
        
        super().__init__()
        
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]
    
    def sample_new_exmps(self, steps=60, step_size=10):
        
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1 
        old_imgs = torch.cat(random.choices(self.examples, k = self.sample_size-n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)
        
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)
        
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        
        return inp_imgs
    
    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True
        
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        
        imgs_per_step = []
        
        for _ in range(steps):
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)
            
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)
            
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)
            
            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())
                
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        
        torch.set_grad_enabled(had_gradients_enabled)
        
        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs
        
class DeepEnergyModel(pl.LightningModule):
    
    def __init__(self, img_shape, batch_size, alpha = 0.1, lr = 1e-4, beta1 = 0.0, **CNN_args):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.cnn = CNNModel(**CNN_args)
        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)
        
    def forward(self, x):
        
        z = self.cnn(x)
        
        return z
    
    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.parameters(), lr = self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        
        real_imgs, _ = batch
        small_noise = torch.rand_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)
        
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)
        
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim = 0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim = 0)
        
        reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss
        
        # Logging
        self.log('loss', loss)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        
        return loss 
    
    def validation_step(self, batch, batch_idx):
        
        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1
        
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim = 0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim = 0)
        
        cdiv = fake_out.mean() - real_out.mean()
        
        self.log('val_contrastive_divergence', cdiv)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())
        
class GenerateCallback(pl.Callback):
    
    def __init__(self, batch_size = 8, vis_steps = 8, num_steps = 256, every_n_epochs = 5):
        
        super().__init__()
        self.batch_size = batch_size
        self.vis_steps = vis_steps
        self.num_steps = num_steps
        self.every_n_epochs = every_n_epochs
        
    def on_epoch_end(self, trainer, pl_module):
        
        if trainer.current_epoch % self.every_n_epochs == 0:
            imgs_per_step = self.generate_imgs(pl_module)
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True)
                #trainer.logger.experiment.add_image(f"generation_{i}", grid, global_step=trainer.current_epoch)
    
    def generate_imgs(self, pl_module):
        
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, 
                                                 step_size=10, return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step
    
class SamplerCallback(pl.Callback):
    
    def __init__(self, num_imgs = 32, every_n_epochs = 5):
        
        super().__init__()
        self.num_imgs = num_imgs
        self.every_n_epochs = every_n_epochs
        
    def on_epoch_end(self, trainer, pl_module):
        
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(random.choice(pl_module.sampler.examples, k = self.num_imgs), dim = 0)
            grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True)
            
class OutlierCallback(pl.Callback):
    
    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size
        
    def on_epoch_end(self, trainer, pl_module):
        
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()
            
        #trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)
        
def train_model(**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         #accelerator="mps",
                         devices=1,
                         max_epochs=10,
                         gradient_clip_val=0.1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_contrastive_divergence"),
                                    GenerateCallback(every_n_epochs=5),
                                    SamplerCallback(every_n_epochs=5),
                                    OutlierCallback(),
                                    LearningRateMonitor("epoch")])
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
    pl.seed_everything(42)
    model = DeepEnergyModel(**kwargs)
    trainer.fit(model, train_loader, test_loader)
    model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    return model
            
##################################################################################################################

model = train_model(img_shape = (1, 28, 28), 
                    batch_size = train_loader.batch_size,
                    lr = 1e-4,
                    beta1=0.0)        
        























































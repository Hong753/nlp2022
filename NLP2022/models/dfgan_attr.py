import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FullyConnectedLayer(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 activation="linear", 
                 lr_multiplier=1, 
                 bias_init=0):
        super().__init__()
        self.activation = activation
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        
    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        
        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        elif self.activation == "lrelu":
            x = torch.addmm(b.unsqueeze(0), x, w.t())
            x = F.leaky_relu(x, 0.2)
        else:
            raise NotImplementedError
        return x

class MappingNetwork(nn.Module):
    def __init__(self, num_attributes, num_layers=8, embed_dim=512, lr_multiplier=0.01):
        super().__init__()
        self.num_attributes = num_attributes
        self.num_layers = num_layers
        self.embed = FullyConnectedLayer(num_attributes, embed_dim)
        for idx in range(num_layers):
            layer = FullyConnectedLayer(embed_dim, embed_dim, activation="lrelu", lr_multiplier=lr_multiplier)
            setattr(self, f"fc{idx}", layer)
        
    def forward(self, c):
        x = self.embed(c)
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)
        return x
        
class Affine(nn.Module):
    def __init__(self, c_dim, num_features):
        super().__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([("fc1", nn.Linear(c_dim, num_features)), 
                                                   ("relu", nn.ReLU(inplace=True)), 
                                                   ("fc2", nn.Linear(num_features, num_features))]))
        self.fc_beta = nn.Sequential(OrderedDict([("fc1", nn.Linear(c_dim, num_features)), 
                                                  ("relu", nn.ReLU(inplace=True)), 
                                                  ("fc2", nn.Linear(num_features, num_features))]))
        self._initialize()
        
    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.fc2.weight.data)
        nn.init.ones_(self.fc_gamma.fc2.bias.data)
        nn.init.zeros_(self.fc_beta.fc2.weight.data)
        nn.init.zeros_(self.fc_beta.fc2.bias.data)
        
    def forward(self, x, y):
        w = self.fc_gamma(y)
        b = self.fc_beta(y)
        
        if w.dim() == 1:
            w = w.unsqueeze(dim=0)
        if b.dim() == 1:
            b = b.unsqueeze(dim=0)
        
        w = w.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(x.size())
        b = b.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(x.size())
        return w * x + b

class FusionBlock(nn.Module):
    def __init__(self, c_dim, in_channels):
        super().__init__()
        self.affine1 = Affine(c_dim, in_channels)
        self.affine2 = Affine(c_dim, in_channels)
        
    def forward(self, x, y):
        h = self.affine1(x, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        return h

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, c_dim, upsample=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c_dim = c_dim
        self.upsample = upsample
        
        self.fuse1 = FusionBlock(c_dim, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fuse2 = FusionBlock(c_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, y):
        if self.upsample is True:
            x = F.interpolate(x, scale_factor=2)
        h = self.fuse1(x, y)
        h = self.conv1(h)
        h = self.fuse2(h, y)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            assert hasattr(self, "shortcut")
            return h + self.shortcut(x)
        else:
            return h + x

class Generator(nn.Module):
    def __init__(self, 
                 z_dim, 
                 c_dim, 
                 img_resolution, 
                 img_channels, 
                 channel_base, 
                 num_attributes):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(3, self.img_resolution_log2 + 1)]
        channel_max = channel_base * 8
        channels_dict = {res: min(channel_base * img_resolution // res, channel_max) for res in self.block_resolutions}
        
        self.z_emb = nn.Linear(z_dim, channel_base * 8 * 4 * 4)
        self.c_emb = MappingNetwork(num_attributes, embed_dim=c_dim)
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 8 else channel_max
            out_channels = channels_dict[res]
            block = GBlock(in_channels, out_channels, z_dim + c_dim, upsample=True)
            setattr(self, f"b{res}", block)

        self.to_rgb = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), 
                                    nn.Conv2d(out_channels, img_channels, kernel_size=3, stride=1, padding=1), 
                                    nn.Tanh())
    
    def forward(self, z, c):
        out = self.z_emb(z)
        out = out.view(z.size(0), -1, 4, 4)
        c = self.c_emb(c)
        c = torch.cat([z, c], dim=1)
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            out = block(out, c)
        out = self.to_rgb(out)
        return out

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        out = self.conv1(x)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.conv2(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        
        if self.in_channels != self.out_channels:
            assert hasattr(self, "shortcut")
            x = self.shortcut(x)
            
        if self.downsample is True:
            x = F.avg_pool2d(x, 2)
            
        return x + self.gamma * out

class Discriminator(nn.Module):
    def __init__(self, 
                 img_resolution, 
                 img_channels, 
                 channel_base):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channel_max = channel_base * 8
        channels_dict = {res: min(channel_base * img_resolution // res, channel_max) for res in self.block_resolutions}
        
        self.img_emb = nn.Conv2d(img_channels, channel_base, kernel_size=3, stride=1, padding=1)
        for res in self.block_resolutions:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2] if res > 8 else channel_max
            block = DBlock(in_channels, out_channels, downsample=True)
            setattr(self, f"b{res}", block)
    
    def forward(self, x):
        out = self.img_emb(x)
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            out = block(out)
        return out

class CondEpilogue(nn.Module):
    def __init__(self, c_dim, channel_base):
        super().__init__()
        self.c_dim = c_dim
        
        self.c_emb = MappingNetwork(312, embed_dim=c_dim)
        self.conv = nn.Sequential(nn.Conv2d(channel_base * 8 + c_dim, channel_base * 2, kernel_size=3, stride=1, padding=1, bias=False), 
                                  nn.LeakyReLU(0.2, inplace=True), 
                                  nn.Conv2d(channel_base * 2, 1, kernel_size=4, stride=1, padding=0, bias=False))
    
    def forward(self, x, y):
        y = self.c_emb(y)
        y = y.view(-1, self.c_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        return torch.flatten(out, start_dim=1)

class DFGAN(nn.Module):
    def __init__(self, G, D, C):
        super().__init__()
        self.G = G
        self.D = D
        self.C = C
        
    @torch.no_grad()
    def _sample_inputs(self, batch_size, device):
        z = torch.randn(batch_size, self.G.z_dim, device=device)
        return z
        
    def forward(self, real_images, c, mode):
        assert mode in ["G_main", "D_main", "D_reg"]
        
        # Sample inputs
        batch_size = real_images.size(0)
        device = real_images.device
        if mode in ["G_main", "D_main"]:
            gen_z = self._sample_inputs(batch_size, device)
            
        # G_main: Maximize logits for generated images
        if mode == "G_main":
            gen_images = self.G(gen_z, c)
            gen_features = self.D(gen_images)
            gen_logits = self.C(gen_features, c)
            g_loss = -gen_logits.mean()
            return g_loss
        
        # D_main
        elif mode == "D_main":
            # Minimize logits for generated images
            with torch.no_grad():
                gen_images = self.G(gen_z, c)
            gen_features = self.D(gen_images)
            gen_logits = self.C(gen_features, c)
            d_loss_gen = F.relu(1.0 + gen_logits).mean()
            
            # Maximize logits for real images
            real_features = self.D(real_images)
            real_logits = self.C(real_features, c)
            d_loss_real = F.relu(1.0 - real_logits).mean()
            
            # Minimize logits for mismatch images
            mis_features = torch.cat([real_features[1:], real_features[0:1]], dim=0)
            mis_logits = self.C(mis_features, c)
            d_loss_mis = F.relu(1.0 + mis_logits).mean()
            return d_loss_gen, d_loss_real, d_loss_mis
        
        # D_reg: Apply MAGP
        elif mode == "D_reg":
            real_images.requires_grad_()
            c.requires_grad_()
            real_features = self.D(real_images)
            real_logits = self.C(real_features, c)
            grads = torch.autograd.grad(outputs=real_logits, 
                                        inputs=[real_images, c], 
                                        grad_outputs=torch.ones(real_logits.size()).to(device), 
                                        retain_graph=True, 
                                        create_graph=True, 
                                        only_inputs=True)
            grad0 = torch.flatten(grads[0], start_dim=1)
            grad1 = torch.flatten(grads[1], start_dim=1)
            grad = torch.cat([grad0, grad1], dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = 2.0 * torch.mean((grad_l2norm) ** 6)
            return d_loss_gp
        

if __name__ == "__main__":
    z_dim = 100
    c_dim = 256
    img_resolution = 256
    img_channels = 3
    channel_base = 32
    
    print("[Generator]")
    generator = Generator(z_dim, c_dim, img_resolution, img_channels, channel_base)
    for res in generator.block_resolutions:
        block = getattr(generator, f"b{res}")
        print(f"res({res}), in({block.in_channels}), out({block.out_channels})")
    
    print("[Discriminator]")
    discriminator = Discriminator(img_resolution, img_channels, channel_base)
    for res in discriminator.block_resolutions:
        block = getattr(discriminator, f"b{res}")
        print(f"res({res}), in({block.in_channels}), out({block.out_channels})")
    
    print("[Conditional Epilogue]")
    cond_epilogue = CondEpilogue(c_dim, channel_base)
    
    batch_size = 3
    z = torch.randn(batch_size, z_dim)
    c = torch.rand(batch_size, c_dim)
    gen_images = generator(z, c)
    
    features = discriminator(gen_images)
    out = cond_epilogue(features, c)
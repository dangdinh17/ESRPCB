import torch
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.utils import *
from models.esrpcb import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esrpcb_c_path = 'outputs/weight_sr/best_esrpcb_canny.pth'

esrpcb_c = ESRPCB(scale_factor=4, use_canny=True)
esrpcb_c.load_state_dict(torch.load(esrpcb_c_path, map_location=device))

lr_path = 'dataset\l_light_01_missing_hole_04_2_150.jpg'
hr_path = 'dataset\l_light_01_missing_hole_04_2_600.jpg'

lr_img = Image.open(lr_path).convert('RGB')
hr_img = Image.open(hr_path).convert('RGB')

transform = transforms.ToTensor()
lr_image = transform(lr_img).unsqueeze(0).to(device)
hr_image = transform(hr_img).unsqueeze(0).to(device)

output_img = esrpcb_c(lr_image)
psnr,ssim = calculate_metrics(output_img, hr_image)
print(f'PSNR: {psnr}, SSIM: {ssim}')

inputs = torch.randn((1, 3, 256, 256)).to(device)
flop, param = summary_model(esrpcb_c, inputs)
print(f'GMac:{flop}, Params:{param}')

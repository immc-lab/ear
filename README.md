
# Erasing Concepts from Unified Autoregressive Models
### [Project Website](https://immc-lab.github.io/ear/) | [Arxiv Preprint](https://arxiv.org/pdf/2506.20151.pdf)  <br>
## Installation Guide
```shell
git clone https://github.com/rohitgandikota/erasing.git
cd erasing
pip install -r requirements.txt
```
## Training Guide
### Janus-pro
After installation, follow these instructions to train a custom EAR model for Janus-pro.:
```
python ear.py --erase_concept 'Nudity' 
```

## Generating Images
Generating images from custom EAR model is super easy. Please follow `notebook/esd_inference_sdxl.ipynb` notebook

For an automated script to generate a ton of images for your evaluations use our evalscripts
```
python evalscripts/ear.py --base_model 'base_model' --esd_path 'esd_path' --num_images 1 --prompts_path 'prompts_path' --num_inference_steps 20 --guidance_scale 7
```

## NOTE ON LICENSE
The code and methods behind our work have been released under IMU. However, the models that you use our methods with, might be on a different licenses. Please read the model's license (the model you are using) carefully for more details.

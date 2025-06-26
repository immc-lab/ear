
# Erasing Concepts from Unified Autoregressive Models
### [Project Website](https://immc-lab.github.io/ear/) | [Arxiv Preprint](https://arxiv.org/pdf/2303.07345.pdf) | [Fine-tuned Weights](https://erasing.baulab.info/weights/esd_models/) | [Demo](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion) <br>
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

## Running Gradio Demo Locally
To run the gradio interactive demo locally, clone the files from [demo repository](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/tree/main) <br>

- Create an environment using the packages included in the requirements.txt file
- Run `python app.py`
- Open the application in browser at `http://127.0.0.1:7860/`
- Train, evaluate, and save models using our method

## NOTE ON LICENSE
The code and methods behind our work have been released under IMU. However, the models that you use our methods with, might be on a different licenses. Please read the model's license (the model you are using) carefully for more details.

## Citing our work
The preprint can be cited as follows
```bibtex
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Jaden Fiotto-Kaufman and David Bau},
  booktitle={Proceedings of the 2023 IEEE International Conference on Computer Vision},
  year={2023}
}
```
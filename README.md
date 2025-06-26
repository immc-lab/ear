
# Erasing Concepts from Unified Autoregressive Models
### [Project Website](https://immc-lab.github.io/ear/) | [Arxiv Preprint](https://arxiv.org/pdf/2506.20151)  <br>
![图片名称](./images/figure_1.png)
## Installation Guide
```shell
git clone https://github.com/immc-lab/ear.git
cd ear
pip install -r requirements.txt
```
## Training Guide
### Janus-pro
After installation, follow these instructions to train a custom EAR model for Janus-pro:

Please run after checking the file path:
```shell
python train/ear_train_church.py 
```

## Generating Images
Generating images from custom EAR model is super easy. Please follow `notebook/esd_inference_sdxl.ipynb` notebook

For an automated script to generate a ton of images for your evaluations use our evalscripts
```shell
python infer/infer_church.py
```

## Evaluation 
You can execute the following command to evaluate the generated data. The specific evaluation method can be found in our paper.
```shell
python eval/eval_object.py  --folder_path {args.output_dir} --topk 10 --batch_size 250
```

## References

This repo is the code for the paper *EAR: Erasing Concepts from Unified Autoregressive Models*.

Thanks for the creative ideas of the pioneer researches:

- https://github.com/rohitgandikota/erasing: **Erasing Concepts from Diffusion Models**
- https://github.com/Con6924/SPM: **One-dimentional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing
Applications**
- https://github.com/koushiksrivats/robust-concept-erasing: **STEREO: A Two-Stage Framework for Adversarially Robust Concept Erasing from Text-to-Image Diffusion Models**
- https://github.com/OPTML-Group/Diffusion-MU-Attack: **To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**
- https://github.com/deepseek-ai/Janus: **Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation**
- https://github.com/deepseek-ai/Janus: **Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling**


## NOTE ON LICENSE
The code and methods behind our work have been released under IMU. However, the models that you use our methods with, might be on a different licenses. Please read the model's license (the model you are using) carefully for more details.

## Citing our work
The preprint can be cited as follows
```bibtex
@misc{fan2025earerasingconceptsunified,
      title={EAR: Erasing Concepts from Unified Autoregressive Models}, 
      author={Haipeng Fan and Shiyuan Zhang and Baohunesitu and Zihang Guo and Huaiwen Zhang},
      year={2025},
      eprint={2506.20151},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.20151}, 
}
```
# Erasing Concepts from Unified Autoregressive Models

### [Project Website](https://immc-lab.github.io/ear/) | [Arxiv Preprint](https://arxiv.org/pdf/2506.20151)  <br>

![图片名称](./images/figure_1.png)

## Installation Guide

### Environment

```shell
git clone https://github.com/immc-lab/ear.git
cd ear
pip install -r requirements.txt
```

### Janus-pro

Ensure that your environment can run Janus-pro, refer to its
official [Quick Start](https://github.com/deepseek-ai/Janus) for details.

## Training Guide

After installation, follow these instructions to train EAR model for Janus-pro.

Please run the script in `train/` after checking the file path:

```shell
python train/ear_train_church.py 
```

## Generating Images with EAR

Image generation using the custom EAR model is a straightforward process. Please run the script in `infer/`.

For automated batch generation of evaluation images, utilize the following script:

```shell
python infer/infer_church.py
```

## Evaluation

You can execute the following command to evaluate the generated data. Please run the script in `eval/`.

The specific evaluation method can be found in our [paper](https://arxiv.org/pdf/2506.20151).

```shell
python eval/eval_object.py  --folder_path {args.output_dir} --topk 10 --batch_size 250
```

## References

This repo is the code for the paper *EAR: Erasing Concepts from Unified Autoregressive Models*.

Thanks for the creative ideas of the pioneer researches:

- https://github.com/rohitgandikota/erasing: **Erasing Concepts from Diffusion Models**
- https://github.com/Con6924/SPM: **One-dimentional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing
  Applications**
- https://github.com/koushiksrivats/robust-concept-erasing: **STEREO: A Two-Stage Framework for Adversarially Robust
  Concept Erasing from Text-to-Image Diffusion Models**
- https://github.com/OPTML-Group/Diffusion-MU-Attack: **To Generate or Not? Safety-Driven Unlearned Diffusion Models Are
  Still Easy To Generate Unsafe Images ... For Now**
- https://github.com/deepseek-ai/Janus: **Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and
  Generation**
- https://github.com/deepseek-ai/Janus: **Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model
  Scaling**

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
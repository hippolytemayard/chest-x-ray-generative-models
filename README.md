# chest-x-ray-ddpm

Implementation of [Denoising Diffusion Probabilistic Model](https://arxiv.org/abs/2006.11239) [1] for chest X-ray image generation. \
The dataset used for this project is an open dataset [2] and is available publicly on [Kaggle](https://www.kaggle.com/datasets/francismon/curated-covid19-chest-xray-dataset).


<img src="./images/diffusion_chest.png" alt="drawing" width="400"/>



This implementation is derived from the following Pytorch implementation [![GitHub](https://i.stack.imgur.com/tskMh.png)GitHub](https://github.com/lucidrains/denoising-diffusion-pytorch)

## Denoising Diffusion Probabilistic Model 


<img src="./images/generated-chest-x-rays.png" alt="drawing" width="400"/>


[1] Denoising Diffusion Probabilistic Model, Ho et al. (2020) \
[2] Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays), Sait et al. (2020)
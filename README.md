# Generative Adversarial Shaders

This project corresponds to the paper "Generative Adversarial Shaders for Real-Time Realism Enhancement".

## Train the pipeline

The required datasets to train the pipeline can be found here:

- Playing For Data (GTA frames): https://download.visinf.tu-darmstadt.de/data/from_games/
- Cityscapes: https://www.cityscapes-dataset.com/
- KITTI: https://www.cvlibs.net/datasets/kitti/
- Mapillary: https://www.mapillary.com/

The files train_shaders.sh and train_shaders_cmap_constraint.sh contain examples on how to launch the pipeline training. For better results quality, we recommend training the pipeline with the cmap constraint applied.

Pytorch Image Classification 

Custom CNN
Epoch 20/20:
Train Loss: 0.1839, Train Acc: 93.87%
Test Loss: 0.2441, Test Acc: 92.02%

ResNet Test Acc: 98.02%

## Dataset

The EuroSAT dataset is a land use and land cover classification dataset based on Sentinel-2 satellite imagery[1][2]. It consists of 27,000 labeled and geo-referenced samples covering 13 spectral bands and 10 different land cover classes[1][3]. The dataset is designed to serve as a benchmark for deep learning applications in Earth observation, particularly for land use and land cover classification tasks[2][4].

Key features of the EuroSAT dataset include:

1. 10 land cover classes
2. 27,000 labeled images
3. Two versions available: RGB (3 bands) and all 13 spectral bands[1]
4. Image size of 64x64 pixels[1]
5. Freely accessible and released under the MIT license[2][3]

The EuroSAT dataset has been widely used in research and has achieved high classification accuracy, with reported results of up to 98.57% overall accuracy using state-of-the-art deep Convolutional Neural Networks (CNNs)[2].

Citations:
- [1] https://www.tensorflow.org/datasets/catalog/eurosat
- [2] https://github.com/phelber/EuroSAT
- [3] https://huggingface.co/datasets/jonathan-roberts1/EuroSAT
- [4] https://www.kaggle.com/code/nilesh789/land-cover-classification-with-eurosat-dataset

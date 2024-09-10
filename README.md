## MCDNet: Multilevel cloud detection network for remote sensing images based on dual-perspective change-guided and multi-scale feature fusion

## Environment 

```
OS: Windows 10
CPU: i9-10850K
GPU: RTX 3080ti
CUDA: 12.1
Python: 3.8.8
Pytorch: 1.13.1
```

## Data

The L8-Biome dataset can be download from [Landsat 8 Cloud Cover Assessment Validation Data | Landsat Missions](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data). 

The WHUS2-CD dataset can be download from [Neooolee/WHUS2-CD: This is a cloud detection validation dataset for Sentinel-2A images](https://github.com/Neooolee/WHUS2-CD). 

Taking the L8-Biome dataset as an example, each image patch used in the train process is list in the `./data/l8/patches.xlsx`. The data is organized like that: 

```
l8
    Train
        cloudy ------------------ cloudy image
        label  ------------------ cloud label
        bccr   ------------------ cloud removal image used bccr
    Test
        cloudy
        label
        bccr
```

Additionally, please note that during training, the label values are assigned as follows: cloudless and no-data areas are labeled as 0, thin clouds are labeled as 1, and thick clouds are labeled as 2.

# Train

You should change the config in the `./utils/config.py`, then input the model name you want to use in the `train.py`, and finally run the model by: 

~~~python
python -u train.py
~~~

# Test

Generally, the model will be saved in the `./data/saved_models/{model-name}` dir, the config will be saved in  the `./data/args/{model-name}` dir, the  evaluation results will be saved in the  `./data/evaluation/{model-name}` dir, and the sample image will be saved in the `./data/show/{model-name}` dir. 

Or you can use `test.py` to evaluate the model's accuracy. 

# Citation

If you use this code for your research, please cite our papers.

~~~
@article{MCDNet,
title = {MCDNet: Multilevel cloud detection network for remote sensing images based on dual-perspective change-guided and multi-scale feature fusion},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {129},
pages = {103820},
year = {2024},
issn = {1569-8432},
doi = {10.1016/j.jag.2024.103820}
}
~~~




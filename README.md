# SRFS- Transformer

# Environment
python ==3.6  
pytorch ==1.80  
opencv-python  
scipy   
h5py   
pillow  
imageio   
nni   
mmcv  
tensorboard  

# Datasets
- Download JHU-CROWD ++ dataset from [here](http://www.crowd-counting.com/)  
- Download NWPU-Crowd dataset (resized) from [Baidu](https://pan.baidu.com/s/1aqiLFU6lo3F_HqeT6wbEjg), password: 04i4 or [Onedrive](https://1drv.ms/u/s!Ak_WZsh5Fl0lhF0V7sxTVv1Vs0Aq?e=drd48k)

# Prepare data
## Generate point map
```cd CLTR/data```  
For JHU-Crowd++ dataset: ```python prepare_jhu.py --data_path /xxx/xxx/jhu_crowd_v2.0```  
For NWPU-Crowd dataset: ```python prepare_nwpu.py --data_path /xxx/xxx/NWPU_CLTR```

## Generate image list
```cd CLTR```    
```python make_npydata.py --jhu_path /xxx/xxx/jhu_crowd_v2.0 --nwpu_path /xxx/xxx/NWPU_CLTR```

# Training 
Example (some hyper-parameters may be different from the original paper):   
```sh experiments/jhu.sh```   
or  
```sh experiments/nwpu.sh```   

* Please change ```nproc_per_node``` and ``` gpu_id``` of ```jhu.sh/nwpu.sh```, if you do not have enogh GPU. 
* We have fixed all random seeds, i.e., different runs will report the same results under the same setting.
* The model will be saved in ```CLTR/save_file/log_file```  
* Note that using FPN will improve the performance, but we do not add it in this version.  
* Turning some hyper-parameters will also bring improvement (e.g., the image size, crop size, number of queries).

# Testing
Example:  
```python test.py --dataset jhu --pre model.pth --gpu_id 2,3```   
or  
```python test.py --dataset nwpu --pre model.pth --gpu_id 0,1``` 

* The model.pth can be obtained from the training phase.

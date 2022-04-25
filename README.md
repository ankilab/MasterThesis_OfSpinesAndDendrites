# Master Thesis: Of Spines And Dendrites

All code used to produce the results and plots in my Master's thesis.

**Docs.** This folder contains the files for generating a Sphinx (https://www.sphinx-doc.org/en/master/) documentation for the repository.

**SIREN.** By importing *SIREN*, the code for performing interplane prediction and motion correction becomes available. The file *network.py* contains the class for SIREN-networks [4]. application.py contains the base class *SIREN_application*, where the network is initialized, and the code for training and prediction shared between all derived application classes is found. The derived classes *InterplanePrediction* and *Motion_Correction* implement the SIREN applications presented in this work. They differ in their preprocessing step. *utils.py* contains functions for generating input and output data for the training procedure from images.
Subsequently, an example for interplane training and prediction is given:

```python
from SIREN import REGISTRY
import tifffile as tif
import numpy as np
# read image stack
img =tif.imread(’./img_1.tif’)
# specify network parameters
args={}
args[’hidden_layers’] = 2
args[’hidden_features’]=128
# initialize network and application
application = REGISTRY[’Interplane’](args)
# convert to 32-bit array
stack = np.asarray(img, dtype=np.float32)
# create data structures for input and output of network
X_train , X_test= plane_int.preprocess(’./’, img)
# train network
application.train(X_train ,n_steps=1000 , steps_to_plot=200)
# prediction for test data
prediction = application.test(X_test)
```

**Thesis_Experiments.** This folder contains the code for the final evaluations and plots contained
in this work.

**deconv.** This package contains the code for all deconvolution techniques applied in this work.
In *deconvolver.py*, the base class is implemented. Each *Deconvolver* has a *preprocess, train, predict* and *predict_img* method. *predict* takes as input a folder name and iterates through all ’.tif’-files and decolvolves them by calling *predict_img*. *BlindRL*, *CAREDeconv* and *Mu_Net* are classes derived from *Deconvolver* implementing specific functionality for the deconvolution techniques. *WBBackProjectorDeconv* implements non-blind Richardson-Lucy with the Wiener-Butterworth filter as back-projector [1]. CAREDeconv uses functionality provided by CSBDeep (http://csbdeep.bioimagecomputing.com/doc/, [3]). The Mu-Net [5] implementation is contained in *mu_net1*. An implementation was provided alongside the publication (https://sites.google.com/view/sehyung/home/projects). For this work, it was migrated from TensorFlow 1 to TensorFlow 2.
An example of how to use this package is given below. It shows how to preprocess the data, train a CARE model and use it for prediction:
```python
from data_augmentation import DataAugmenter 
from deconv import REGISTRY
import tifffile as tif
# initialize hyperparameters
args = {}
args[’z_shape’] = 16
args[’xy_shape’] = 32
args[’n_patches’] = 50
args[’epochs’] = 50
args[’data_path’] = ’./Input_data’
args[’result_path’] = ’./Output’
args[’learning_rate’] = 0.0004
args[’unet_residual’] = True
args[’batch_size’] = 16
# generate training data
data_augmenter = DataAugmenter(args)
_, data_dir = data_augmenter.augment(data_path , source_folder ,
target_folder , care=True ,
data_dir=data_augmented_path)
# initialize and train CARE model
deconvolver = REGISTRY[’csbdeep’](args)
model_dir , mdl_path = deconvolver.train(args[’data_path’], args[’
epochs’], args[’batch_size’],
args[’learning_rate’], args[’
unet_residual’])
# predict test data
deconvolver.predict(’./Test_data’, model_dir , model_path , args[’
result_path’])
```

**denoising.** This package contains the code for the denoising techniques using Gaussian filter,
Bilateral filter and Neighbor2Neighbor [2]. *denoiser.py* comprises the code for the super-class
*Denoiser* and all implemented derived denoising techniques. Each *Denoiser* has a *prepare*
and *denoise* function. *Neighbor2Neighbor* is trained by calling *prepare()*. *denoise()* predicts
the undistributed version of the input image. The Neighbor2Neighbor-folder contains
the code for for network training and prediction. It was adjusted from the implementation provided
alongside the publication (https://github.com/TaoHuang2018/Neighbor2Neighbor). Exemplary usage is shown below:
```python
import tifffile as tif
import utils
from denoising import REGISTRY
# the network is implemented for 2D data - thus , 3D stack has to be
split up in 2D images
file=tif.imread(’./img_1.tif’)
utils.save_tif_stack_to_2d(file , ’Img_2D’, name=’img_1.tif’)
# initialize Neighbor2Neighbor network
denoiser = REGISTRY[’n2n’]()
# train network and store training history in result folder
# for simplification , training data is same as validation data
denoiser.prepare(’./Img_2D’,’./Img_2D’, ’./Results’)
# denoise image using trained network
test_file=tif.imread(’./img_2.tif’)
denoised=denoiser.denoise(test_file)
```

**Train_Test.csv.** This file lists all image stacks considered for training. For each region and mouse, one timestamp is included. Some preprocessing functionality contained in *utils.py* is based on this file. By marking stacks with ’X’, it is indicated that these stacks need to be
processed.

**UI_Test_img.py.** A Graphical User Interface is launched by executing this file, showing a randomly selected image plane deconvolved using two different approaches. The user is asked to select the preferred image plane. Upon completion, statistics reflecting on the preferred
deconvolution technique are returned.

**config.yaml.** The settings for deconvolution and denoising can be specified in this file instead of specifying the hyperparameters like in the implementation examples shown above.

**data_augmentation.py.** This file contains the classes *DataAugmenter* and *DataProvider*. *DataAugmenter* creates patch pairs from files in a given folder and stores them. This is important for training CARE and Mu-Net. *DataProvider* loads requested subsets from file using flammkuchen (https://github.com/portugueslab/flammkuchen). It is used as data loader during Mu-Net training. Th example below shows how to use DataProvider:
```python
from data_augmentation import DataProvider
# initializes data provider
# calls DataAugmenter internally and generates patches
data_provider = DataProvider((16, 32), ’./Data’, ’Raw’,
’Restored’, n_patches=50)
# shuffles data
data_provider.shuffle()
# get patch -pairs
input , output = self.data_provider.get(batch_size=16)
```

**imagequalitymetrics.py.** This file contains the *ImageQualityMetrics* class. After initalizing an object, metrics like SSIM, MSE and PSNR can be computed.

**normalize.py.** The base class Normalizer has only a normalize-function. The derived classes *PercentileNormalizer*, *MinMaxNormalizer*, and *Rescaler* implement this function according to their definition.

**registration.py.** By executing this file, all files specified in Train_Test.csv are registered and saved. Source and destination path need to be specified. (Using PyStackReg: https://pystackreg.readthedocs.io/en/latest/)

**utils.py.** This file contains various functions for data handling.




[1] Min Guo et al. “Rapid image deconvolution and multiview fusion for optical microscopy”.
In: Nature Biotechnology 38.11 (2020), pp. 1337–1346. issn: 15461696. url: https://doi.org/10.1038/s41587-020-0560-x.

[2] Tao Huang et al. “Neighbor2Neighbor: Self-Supervised Denoising From Single Noisy Images”. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021. Computer Vision Foundation / IEEE, 2021, pp. 14781–14790.

[3] Martin Weigert et al. “Content-aware image restoration: pushing the limits of fluorescence microscopy”. In: Nature Methods 15.12 (2018), pp. 1090–1097. issn: 15487105. doi: 10.1038/s41592-018-0216-7.

[4] Vincent Sitzmann et al. “Implicit Neural Representations with Periodic Activation Functions”.
In: Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020,
virtual. Ed. by Hugo Larochelle et al. 2020. url: https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html.

[5] Sehyung Lee et al. “Mu-net: Multi-scale U-net for two-photon microscopy image denoising and restoration”. In: Neural Networks 125 (2020), pp. 92–103. doi: 10.1016/j.neunet.2020.01.026.

# Local Binary Feature Compression Libary
# 1. About


**Summary:**  
This repository contains a public version of the feature compression library in version 2 (LBFC2) used for compression of ORB features for collaborative remote SLAM. It is capable of compressiong ORB features from monocular and stereo sensors. Additionally, the depth can be estimated using stereo-matching and transmitted as (quantized) side information. A version of this framework was used in **Collaborative Visual SLAM using Compressed Feature Exchange**. 

Although, it is currently written for ORB features, its modular structure encapsulating all compression steps into individual functions makes it easy to replace individual coding modes or the binary descriptor. More general, this framework can also be used for applications besides remote visual SLAM. 

**Authors:**  
Dominik Van Opdenbosch (dominik dot van-opdenbosch at tum dot de) and Eckehard Steinach   
Chair of Media Technology, Technical University of Munich, 2018


# 2. Related Publications


[1] **A Joint Compression Scheme for Local Binary Feature Descriptors and their Corresponding Bag-of-Words Representation**  
D. Van Opdenbosch, M. Oelsch, A. Garcea, and E. Steinbach  
*IEEE Visual Communications and Image Processing (VCIP),* 2017. 

[2] **Efficient Map Compression for Collaborative Visual SLAM**  
D. Van Opdenbosch, T. Aykut, M. Oelsch, N. Alt, and E. Steinbach  
*IEEE Winter Conference on Applications of Computer Vision (WACV),* 2018. 

[3] **Selection and Compression of Local Binary Features for Remote Visual SLAM**  
D. Van Opdenbosch, and E. Steinbach  
*IEEE International Conference on Image Processing (ICIP),* 2018. 

[4] **Collaborative Visual SLAM using Compressed Feature Exchange**  
D. Van Opdenbosch, and E. Steinbach  
*IEEE Robotics and Automation Letters,* 2018. 

[5] **Flexible Rate Allocation for Binary Feature Compression**  
D. Van Opdenbosch, M. Oelsch, A. Garcea, and E. Steinbach  
*IEEE Visual Communications and Image Processing (VCIP),* 2018. 

# 3. License
The feature compression library is released under a [GPLv3 license](https://www.gnu.org/licenses/gpl.html). A list of known code dependencies with their respective licenses is noted in the Dependencies.md file. 

If you use our approach in an academic work, please cite:

	@article{VanOpdenbosch2018,
		author = {{Van Opdenbosch}, Dominik and Steinbach, Eckehard},
		journal = {IEEE Robotics and Automation Letters (RAL)},
		title = {{Collaborative Visual SLAM using Compressed Feature Exchange}},
		year = {2018}
	}



# 4. Prerequisites

**This is research code.** We think that the code might be useful for other research groups, but we want to emphasize that the code comes *without any warranty* and *no guarantee* to work in your setup. In consequence we can not promise *any support*. However, we would be glad to hear from you if you use this code, fix bugs or improve the implementation.


**Turn on optimizations.** Make sure that everything is compiled in `release` mode with all optimizations `-O3` and `-march=native` activated. Turn off the viewer if not needed as this requires additional resources. 


We have tested the library using Ubuntu **14.04, 16.04**, but it should be easy to compile on other platforms. 

# 5. Installation 

We provide a script `build.sh` to build the *Thirdparty* libraries and *ORB-SLAM2*.  

```
cd featureCompression
mkdir build
mkdir install
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)/../install
make -j 2 install
```


# 6. Run the examples

We do not ship any specific examples in the library, but provide examples in the collaborative visual SLAM framework. 


# 7. FAQ: 

**1) Check the coding parameters:**  

Currently, the encoder configuration such as the coding modes, number of reference frames etc. is **not** signaled in the bitstream. This means, the decoder has to be manually configured to the same number of reference frames, otherwise it might crash. 


**2) Feature selection is not available:**

Due to the restructuring of the library to support multi-threaded feature encoding, the support for feature selection was dropped. It is implemented in v1 of the library (available [here](https://d-vo.github.io/ICRA18/)). The rate allocation method introduced in [5] is not implemented in any public version. 


**3) What is inside the .vstats file:**

The .vstats file contains the trained probabilities for the residual coding. It also containes additional information that is used in different versions of the coding framework such as the probabilities for the feature selection (v1). 

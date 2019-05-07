This computer program is for image reconstruction from PET coincidence data. The input of coincidence data should be in binary format. The outputs are reconstructed images at all iterations. The output image is also in binary format, which could be displayed using AMIDE. The order of the indices to traverse the image is that the x-axis is the fastest running index, followed by y, then z. The binary image file could also be read using C++ fstream library. The program will also output the value of objective function that is maximized during reconstruction at every iteration, which is stored in a file called "ObjectiveFuncValue.txt".

Image reconstructions are performed using the maximum-likelihood expectation maximization (MLEM) algorithm and the orthogonal distance-based ray-tracer (OD-RT) as the geometrical projector (P. Aguiar, et al. Med. Phys., vol. 37, no. 11, pp. 5691-5702, Nov. 2010.). The image reconstruction program has GPU acceleration based on CUDA C, which is essential for fast image reconstruction.

The program supports normalization using image or coincidence data, which could be configured using the "configRecon.txt" file. The coincidence data is from PET scan of a normalization phantom. The normalization phantom has the same attenuation as the phantom for imaging study, but has uniform activity distribution across the whole FOV. If coincidence data is used for normalization, an image showing the sensitivity distribution across the whole FOV will be output with name being "normImage". When reconstructing images for imaging study, you could directly use the "normImage" for normalization.

The program also supports penalized maximum-likelihood (PML) image reconstruction. The purpose is to mitigate limited-angle artifacts in the image reconstruction for a PET system that does not have full angular coverage of the imaging plane. The regularization term penalizes the dissimilarity between the target image and a whole-body image of the same object. You could also choose to incorporate an image-based resolution model in the regularization term to improve image quality by changing the "configRecon.txt" file.

Image-based resolution model for MLEM image reconstruction is also available, but is currently under development. You could use it by changing the value of "ImagePSF" to 1 or 2 in the "configRecon.txt" file. You could also extend the functionality by adding additional resolution models.

The program has been tested on the operating system Ubuntu 16.04 with NVIDIA CUDA Toolkit 9.0 installed. The source code of the program is in the following files:
	"imagerecon.cu": contains main function.
	"*.h": contain function, class, structure, and global variable definition.
The source code is compiled using NVIDIA's nvcc compiler which is part of NVIDIA CUDA Toolkit. To compile the code, simply type "make" in a terminal.  

To run the image reconstruction code, use the following command:
    ./imagerecon coincidenceFileName outputImageName regularizationImageName normalizationFileName
Substitute the names in the command with the real name of each file. If you do not want to use regularization or normalization, simply use "non" for the file name and set the value of relevant varialbes in "configRecon.txt" file accordingly. The "configRecon.txt" file should be present in the directory where the reconstruction program is running.






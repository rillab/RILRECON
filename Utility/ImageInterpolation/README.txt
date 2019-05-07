The computer program is for trilinear interpolation of a whole-body PET image in order to match the voxel size of a dedicated PET image. The input image and output image are both in binary format. The order of the indices to traverse the image is that the x-axis is the fastest running index, followed by y, then z.

To compile the code, simply type "make" in a terminal. Compiler: g++ 4.7.3.

The command for executing the code is as follows:
    ./trilinear inputFile
Then you will be asked to enter FOV size (unit: mm) in x,y,z direction, the voxel size for input image, and the voxel size for output image. An image with the new voxel size named "inputFile.normal" will be output. 


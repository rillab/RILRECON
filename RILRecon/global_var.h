#ifndef global_var_H
#define global_var_H

#include <string>
#include <cuda_runtime.h>
#include "device_functions.h"

#include "lor.h"
#include "timing.h"

//////////////////////////////////////
// Declare global variables.
//////////////////////////////////////

float a = 4.; //voxel size for reconstructed image.
float bndry[3] = {200., 200., 216.};    //FOV x, y, z. Unit: mm.
int msize;  // total number of voxels in image
float torhw, torsgm;  // TOR half width, TOR sigma. They are used in OD-RT geometrical projector.
float beta;  // regularization strength.
int rgl;	//indicator for regularization. 0: no regularization, 1: regularization.
int blur;	//indicator for blurring in regularization. 0: no blurring, 1: spatial-invariant blurring, 3: spatial-variant blurring.
int norma; //indicator for normalization. 0: no normalization, 1: normalization using data and generate sensitivity image with same weight, 2: normalization using sensitivity image, 3: normalization using data and generate sensitivity image with different weight.
float ThreshNorm;	//threshold for normalization
float bsgm[3], bthresh=0.01;	//sigma x,y,z in spatial-invariant blurring Gaussian function for regularization.
float rads[3];	//radius for Gaussian blurring for spatial-invariant regularization.
int indr[3];	//number of voxels for the radius in spatial-invariant Gaussian blurring for regularization.
string blurParaFile;	//file name for spatial-variant blurring for regularization.
int blurParaNum[2];		//number of parameters for spatial-variant blurring for regularization.

float *smatrix;  //array for storing target image
float *snmatrix; //array for storing target image
float *poimage; //array for storing prior image
float *normimage, *dev_normimage;	//normalization image (sensitivity image)
float weight_scale_1 = 1.0, weight_scale_2 = 1.0;	// used for spatial invariant blurring for regularization. 1: two-panel. 2: wb

cudalor xlor, ylor, zlor, dev_xlor, dev_ylor, dev_zlor;  // variables for storing LOR data.

int numline = 0;    //number of lines/lors in each input file.
int *nummainaxis;   //number of lines in each main axis (x,y,z)
int imagepsf = 0;	//whether using image-based PSF
float psfsgm[3], psfrads[3];	//Sigma for spatial invariant PSF
int psfindr[3];  // number of voxels for the radius in spatial-invariant PSF.
string maskFileName;  // mask file name. Use when imagepsf==2.
int psfVoxelSize[3];	// Size of mask image. Use when imagepsf==2.

// variables for timing
TimingCPU timing;
TimingGPU timinggpu;
timevar timeall;



//////////////////////////////////////
// Declare variables in GPU constant memory.
//////////////////////////////////////

__device__ __constant__ float aves[1] , avep[1], d_bsgm[3], d_rads[3], d_info[4];
__device__ __constant__ int d_indr[3], d_imageindex[4], d_lorindex[3];


#endif

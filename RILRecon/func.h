#ifndef func_H
#define func_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cuda_runtime.h>
#include "device_functions.h"

#include "projection.h"
#include "global_var.h"
#include "lor.h"
#include "macro.h"


///////////////////////////////////////////////////////////
// structures declaration for different types of blurring.
///////////////////////////////////////////////////////////
struct Blur_Gaussian_Invariant { __device__ float blur(int x, int y, int z, int ii, int jj, int kk, float a, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff) ;};
struct Blur_Mask_Invariant { __device__ float blur(int x, int y, int z, int ii, int jj, int kk, float a, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff); };
struct Blur_Gaussian_Variant {__device__ float blur(int x, int y, int z, int ii, int jj, int kk, float a, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff); };




//////////////////////////////////////
// Declare functions.
//////////////////////////////////////


void get_normal_image(string filenorm, int nx, int ny, int nz);
vector<string> explode(string s, char c);
float total_weight(int *indr, float *sgm);
float total_weight_image(float *image, int size);
__global__ void total_weight_variant(float * __restrict__ allweight, const float * __restrict__ blurpara, const int numpara, const int numcoeff);
void blur_wrapper(float *image, float *bimage, float *weight, float *sgm, float *rads, int *indr, float weight_scale);
void blur_wrapper(float *image, float *bimage, float *weight, int *psfsize, float *d_psfimage, float weight_scale);
void blur_wrapper(float *image, float *bimage, float *weight, int *indr, const float * __restrict__ dev_blurpara, const int numpara, const int numcoeff);
void proj_wrapper(cudalor dev_xlor, cudalor dev_ylor, cudalor dev_zlor, float *dev_smatrix, float *dev_snmatrix);
int preplor(string fin);
__global__ void calnewmatrix000(float *snmatrix, float *smatrix);
__global__ void calnewmatrix100(float *snmatrix, float *smatrix, float *normimage);
__global__ void calnewmatrix010(float *snmatrix, float *smatrix, float *poimage);
__global__ void calnewmatrix110(float *snmatrix, float *smatrix, float *normimage, float *poimage);
template<typename Operation>
__global__ void calnewmatrix011(float *snmatrix, float *smatrix, float *poimage, float *bmatrix, float *allweight, float weight_scale, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff);
template<typename Operation>
__global__ void calnewmatrix111(float *snmatrix, float *smatrix, float *normimage, float *poimage, float *bmatrix, float *allweight, float weight_scale, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff);
__global__ void calave(float *smatrix, float *gave);
__device__ float eval_quad(const float * __restrict__ blurpara, float r);
template<typename Operation>
__global__ void gpublur(float *smatrix, float *bmatrix, float *allweight, float *psfimage, float weight_scale, const float * __restrict__ blurpara, const int numpara, const int numcoeff);	
__global__ void calLogLike(float *xlinevalue, double *gloglike, const int lorindex);	
__global__ void calLogLikeS(float *smatrix, float *normimage, double *gloglike, const int msize, const int norma);
__global__ void calLogR(float *smatrix, float *poimage, double *gloglike, const int msize);
template<class T>
void readBinary(string filename, T * &data, int size);
template<class T>
void writeBinary(string filename, T * data, int size);
template<class T>
void readTxt(string filename, T * &data, int size);




////////////////////////////////////////
// functions definition.
////////////////////////////////////////

// function for GPU error check.
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// calculate the normalization image using coincidence data from normalization simulation or get the image from file.
void get_normal_image(string filenorm, int nx, int ny, int nz) {
	dim3 threads(threadsperBlock, threadsperBlock);

	if(norma == 2)	//read sensitivity image from input file.
	{
		cout<<"Reading normalization image......"<<endl;
		readBinary<float> (filenorm, normimage, msize);
		cout<<"Finish reading normalization image."<<endl;    

		timinggpu.StartCounter();
		cudaMalloc((void**) &dev_normimage, msize*sizeof(float) );
		cudaMemcpy(dev_normimage, normimage, msize*sizeof(float), cudaMemcpyHostToDevice);
		timeall.memoryIO += timinggpu.GetCounter();

	}


	else if(norma == 3)	//generate sensitivity image from coincidence data in normalization run.
	{
		cout<<"Sorting LORs for normalization and copying to device memory......"<<endl;
		preplor(filenorm); //  read lors in the file, sort lors, copy to cud
        cout<<"Normalization: Number of LORs in each main axis (x,y,x): "<<nummainaxis[0]<<" "<<nummainaxis[1]<<" "<<nummainaxis[2]<<endl;

		cudaMemcpyToSymbol(d_lorindex, nummainaxis, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
		
		normimage = (float*) malloc(msize * sizeof(float));
		cudaMalloc((void**) &dev_normimage, msize*sizeof(float) );
		cudaMemset( dev_normimage, 0, msize*sizeof(float)); 

		cudaMemset( dev_xlor.linevalue, 0, nummainaxis[0]*sizeof(float));	// improved normalization method
		cudaMemset( dev_ylor.linevalue, 0, nummainaxis[1]*sizeof(float));
		cudaMemset( dev_zlor.linevalue, 0, nummainaxis[2]*sizeof(float));
		float *normphantom = (float*) malloc(msize * sizeof(float)), *dev_normphantom;
		cudaMalloc((void**) &dev_normphantom, msize * sizeof(float));
		for(int i=0; i< nx*ny*nz; i++){
			normphantom[i] = 0.001f;
		}
		cudaMemcpy(dev_normphantom, normphantom, msize*sizeof(float), cudaMemcpyHostToDevice);
		
		xfpro<<<blocksPerGrid, threads>>>(dev_xlor, dev_normphantom);
		yfpro<<<blocksPerGrid, threads>>>(dev_ylor, dev_normphantom);
		zfpro<<<blocksPerGrid, threads>>>(dev_zlor, dev_normphantom);
		xbpro<<<blocksPerGrid, threads>>>(dev_xlor, dev_normimage);
		ybpro<<<blocksPerGrid, threads>>>(dev_ylor, dev_normimage);
		zbpro<<<blocksPerGrid, threads>>>(dev_zlor, dev_normimage);

		timinggpu.StartCounter();
		cudaMemcpy(normimage, dev_normimage, msize*sizeof(float), cudaMemcpyDeviceToHost);
		timeall.memoryIO += timinggpu.GetCounter();

        free(normphantom);
        cudaFree(dev_normphantom);
		
		writeBinary<float> ("normImage", normimage, msize);
		cout<<"Finish creating normalization image."<<endl;

	}

	else if(norma == 0) {}
	else cout<<"Unkown indicator for normalization option!!"<<endl;
}


// split a string s to multiple strings (delimiter: c).
vector<string> explode(string s, char c)
{
	string buff="";
	vector<string> v;
	char n;
	
	for(unsigned i=0; i<s.length(); ++i)
	{
		n=s.at(i);
		if(n != c) buff+=n; else
		if(n == c && buff != "") { v.push_back(buff); buff = ""; }
	}
	if(buff != "") v.push_back(buff);
	
	return v;
}


// Calculate weight_scale for spatial-invariant Gaussian blurring.
float total_weight(int *indr, float *sgm)
{
    float weight_scale = 0.0f;
    for (int i = -indr[0]; i <= indr[0]; i++ )
        for (int j = -indr[1]; j <= indr[1]; j++)
            for (int k = -indr[2]; k <= indr[2]; k++)
                weight_scale += exp(-(pow(i,2)/(2.0f * pow(sgm[0],2)) + pow(j,2)/(2.0f * pow(sgm[1],2)) + pow(k,2)/(2.0f * pow(sgm[2],2))) * pow(a,2));

	return weight_scale;

}


// Calculate weight_scale for image-based PSF which is obtained from a reconstructed image of a point source (mask image). 
float total_weight_image(float *image, int size)
{
	float weight_scale = 0.0f;
	for(int i = 0; i < size; i++) weight_scale += image[i];
	return weight_scale;
}


// Calculate all weight_scale for spatial-variant Gaussian blurring.
__global__ void total_weight_variant(float * __restrict__ allweight, const float * __restrict__ blurpara, const int numpara, const int numcoeff)
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];

	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;
	int gz = blockIdx.z * blockDim.z + threadIdx.z;

	float wi = 0.0f;
	if(gx < nx && gy < ny && gz < nz) {
	for(int k = (-1) * d_indr[2]; k <= d_indr[2]; k++ )
		for(int j = (-1)*d_indr[1]; j <= d_indr[1]; j++)
			for(int i = (-1)*d_indr[0]; i <= d_indr[0]; i++)
				wi += Blur_Gaussian_Variant().blur(gx+i,gy+j,gz+k,gx,gy,gz,d_info[0],NULL, blurpara, numpara, numcoeff);

	allweight[gx + gy * nx + gz * nx * ny ] = wi;
	
	}
}

// Blur using Gaussian function
void blur_wrapper(float *image, float *bimage, float *weight, float *sgm, float *rads, int *indr, float scale)
{
	dim3 threads(threadsperBlock, threadsperBlock);
	cudaMemcpyToSymbol(d_bsgm, sgm, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_rads, rads, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_indr, indr, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);

	timinggpu.StartCounter();
	gpublur<Blur_Gaussian_Invariant><<<blocksPerGrid, threads>>>(image, bimage, weight, NULL, scale, NULL, 0,0);
	timeall.tpostimageprocess += timinggpu.GetCounter();

}

// Blur using mask image
void blur_wrapper(float *image, float *bimage, float *weight, int *psfsize, float *d_psfimage, float scale)
{
	dim3 threads(threadsperBlock, threadsperBlock);
	int halfsize[3];
	for(int i=0; i < 3; i++) halfsize[i] = (psfsize[i]-1)/2;
	cudaMemcpyToSymbol(d_indr, halfsize, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);

	timinggpu.StartCounter();
	gpublur<Blur_Mask_Invariant><<<blocksPerGrid, threads>>>(image, bimage, weight, d_psfimage, scale, NULL,0,0);
	timeall.tpostimageprocess += timinggpu.GetCounter();


}

// Blur using spatial-variant Gaussian function
void blur_wrapper(float *image, float *bimage, float *weight, int *indr, const float * __restrict__ dev_blurpara, const int numpara, const int numcoeff) 
{
	dim3 threads(threadsperBlock, threadsperBlock);
	cudaMemcpyToSymbol(d_indr, indr, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);

    timinggpu.StartCounter();
	gpublur<Blur_Gaussian_Variant><<<blocksPerGrid, threads>>>(image, bimage, weight, NULL, 0, dev_blurpara, numpara, numcoeff);
	timeall.tpostimageprocess += timinggpu.GetCounter();

}


// forward and back projection for one iteration in image reconstruction.
void proj_wrapper(cudalor dev_xlor, cudalor dev_ylor, cudalor dev_zlor, float *dev_smatrix, float *dev_snmatrix)
{

	dim3 threads(threadsperBlock, threadsperBlock);

    timinggpu.StartCounter();
    xfpro<<<blocksPerGrid, threads>>>(dev_xlor, dev_smatrix);
    timeall.txforward += timinggpu.GetCounter();

    timinggpu.StartCounter();
    yfpro<<<blocksPerGrid, threads>>>(dev_ylor, dev_smatrix);
    timeall.tyforward += timinggpu.GetCounter();

    timinggpu.StartCounter();
    zfpro<<<blocksPerGrid, threads>>>(dev_zlor, dev_smatrix);
    timeall.tzforward += timinggpu.GetCounter();

    timinggpu.StartCounter();
    xbpro<<<blocksPerGrid, threads>>>(dev_xlor, dev_snmatrix);
    timeall.txbackward += timinggpu.GetCounter();

    timinggpu.StartCounter();
    ybpro<<<blocksPerGrid, threads>>>(dev_ylor, dev_snmatrix);
    timeall.tybackward += timinggpu.GetCounter();

    timinggpu.StartCounter();
    zbpro<<<blocksPerGrid, threads>>>(dev_zlor, dev_snmatrix);
    timeall.tzbackward += timinggpu.GetCounter();

}


// read image file in binary format.
template<class T>
void readBinary(string filename, T * &data, int size) {
	ifstream fin;
	fin.open(filename.c_str(), ios::in | ios::binary);
    if (fin.is_open()){
        data = (T*) malloc(size * sizeof(T));                                                                                                                    
        for(int iii=0; iii< size; iii++) fin.read( (char*)&data[iii], sizeof(T));
    }
    else cout<<"Unable to open input file!!"<<endl;
    fin.close(); 
}


// write image file in binary format.
template<class T>
void writeBinary(string filename, T * data, int size) {
    ofstream fout;
    fout.open(filename.c_str(), ios::out | ios::binary);
    if (fout.is_open()){
        for(int iii=0; iii< size; iii++) fout.write( (char*)&data[iii], sizeof(T));
    }
    else cout<<"Unable to write image to file!!"<<endl;

    fout.close();

}


// read file in txt format. The file only has one line with data separated by space.
template<class T>
void readTxt(string filename, T * &data, int size) {
	 ifstream fin;
	 fin.open(filename.c_str(), ifstream::in);
	 string line;
	 vector<string> num;
	 stringstream ss;

	 if (fin.is_open()) {
		 data = (T*) malloc(size * sizeof(T));
		 while(getline(fin, line)) {
			 num = explode(line, ' ');
			 for(int i=0; i<size; i++) {ss<<num[i];ss>>data[i];ss.clear();}
		 }
	 }
	 else cout<<"Unable to open input file!!"<<endl;
	 fin.close();

}


//function that read lor from fin, sort lor, and copy lor to cuda
int preplor(string filein)
{
	ifstream fin;
	fin.open(filein.c_str(), ios::in | ios::binary);

	vector<lor> alllor;		//matrix for all lor
	numline = 0;
	string line;
	
    // number of LORs in each main axis (x,y,z).
	nummainaxis[0] = 0;
	nummainaxis[1] = 0;
	nummainaxis[2] = 0;

    // read LOR file in binary format. Decide which main axis each LOR belongs to.
	timing.StartCounter();
	if (fin.is_open()){

		while ( !fin.eof() )
		{
			numline += 1;
			float coordlor[6];
			fin.read((char*)coordlor, 6 * sizeof(float));
			lor bufflor;

			bufflor.weight = 1.0;
			bufflor.x1 = coordlor[0] - (-bndry[0]/2. + 0.5 * a);
			bufflor.y1 = coordlor[1] - (-bndry[1]/2. + 0.5 * a);
			bufflor.z1 = coordlor[2] - (-bndry[2]/2. + 0.5 * a);
			bufflor.x2 = coordlor[3] - (-bndry[0]/2. + 0.5 * a);
			bufflor.y2 = coordlor[4] - (-bndry[1]/2. + 0.5 * a);
			bufflor.z2 = coordlor[5] - (-bndry[2]/2. + 0.5 * a);
	
			if(abs(bufflor.x1-bufflor.x2) >= abs(bufflor.y1-bufflor.y2) && abs(bufflor.x1-bufflor.x2) >= abs(bufflor.z1-bufflor.z2)) {bufflor.mainaxis = 0; nummainaxis[0] += 1;}
			else if(abs(bufflor.y1-bufflor.y2) >= abs(bufflor.x1-bufflor.x2) && abs(bufflor.y1-bufflor.y2) >= abs(bufflor.z1-bufflor.z2)) {bufflor.mainaxis = 1; nummainaxis[1] += 1;}
			else if(abs(bufflor.z1-bufflor.z2) >= abs(bufflor.x1-bufflor.x2) && abs(bufflor.z1-bufflor.z2) >= abs(bufflor.y1-bufflor.y2)) {bufflor.mainaxis = 2; nummainaxis[2] += 1;}
			else cout<<"Cannot fing the main axis!!"<<endl;
	
			alllor.push_back(bufflor);
		}
	}
	else cout<<"Unable to open input lor file!!"<<endl;

	fin.close();
	timeall.lorsorting += timing.GetCounter();

    // allocate host memory for storing LORs in each main axis.
	timing.StartCounter();
	xlor.x1 = (float*) malloc(nummainaxis[0] * sizeof(float)); 
	xlor.y1 = (float*) malloc(nummainaxis[0] * sizeof(float));
	xlor.z1 = (float*) malloc(nummainaxis[0] * sizeof(float));
	xlor.x2 = (float*) malloc(nummainaxis[0] * sizeof(float));
	xlor.y2 = (float*) malloc(nummainaxis[0] * sizeof(float));
	xlor.z2 = (float*) malloc(nummainaxis[0] * sizeof(float));
	xlor.linevalue = (float*) malloc(nummainaxis[0] * sizeof(float));
	xlor.weight = (float*) malloc(nummainaxis[0] * sizeof(float));

    ylor.x1 = (float*) malloc(nummainaxis[1] * sizeof(float)); 
    ylor.y1 = (float*) malloc(nummainaxis[1] * sizeof(float));
    ylor.z1 = (float*) malloc(nummainaxis[1] * sizeof(float));
    ylor.x2 = (float*) malloc(nummainaxis[1] * sizeof(float));
    ylor.y2 = (float*) malloc(nummainaxis[1] * sizeof(float));
    ylor.z2 = (float*) malloc(nummainaxis[1] * sizeof(float));
	ylor.linevalue = (float*) malloc(nummainaxis[1] * sizeof(float));
	ylor.weight = (float*) malloc(nummainaxis[1] * sizeof(float));

    zlor.x1 = (float*) malloc(nummainaxis[2] * sizeof(float)); 
    zlor.y1 = (float*) malloc(nummainaxis[2] * sizeof(float));
    zlor.z1 = (float*) malloc(nummainaxis[2] * sizeof(float));
    zlor.x2 = (float*) malloc(nummainaxis[2] * sizeof(float));
    zlor.y2 = (float*) malloc(nummainaxis[2] * sizeof(float));
    zlor.z2 = (float*) malloc(nummainaxis[2] * sizeof(float));
	zlor.linevalue = (float*) malloc(nummainaxis[2] * sizeof(float));
	zlor.weight = (float*) malloc(nummainaxis[2] * sizeof(float));
	timeall.memoryIO += timing.GetCounter();

    // store LORs data into xlor, ylor, zlor.
	timing.StartCounter();
	int cma[3] = {0,0,0};	 
	for(int i=0; i< numline; i++)
	{
		lor bufflor = alllor[i];
		if(bufflor.mainaxis == 0) 
		{
			xlor.x1[cma[0]] = bufflor.x1;
			xlor.y1[cma[0]] = bufflor.y1;
			xlor.z1[cma[0]] = bufflor.z1;
			xlor.x2[cma[0]] = bufflor.x2;
			xlor.y2[cma[0]] = bufflor.y2;
			xlor.z2[cma[0]] = bufflor.z2;
			xlor.weight[cma[0]] = bufflor.weight;
			cma[0] += 1;
		}
		else if(bufflor.mainaxis == 1)
		{
            ylor.x1[cma[1]] = bufflor.x1;
            ylor.y1[cma[1]] = bufflor.y1;
            ylor.z1[cma[1]] = bufflor.z1;
            ylor.x2[cma[1]] = bufflor.x2;
            ylor.y2[cma[1]] = bufflor.y2;
            ylor.z2[cma[1]] = bufflor.z2;
			ylor.weight[cma[1]] = bufflor.weight;

            cma[1] += 1;
	
		}
		else if(bufflor.mainaxis == 2)
		{
            zlor.x1[cma[2]] = bufflor.x1;
            zlor.y1[cma[2]] = bufflor.y1;
            zlor.z1[cma[2]] = bufflor.z1;
            zlor.x2[cma[2]] = bufflor.x2;
            zlor.y2[cma[2]] = bufflor.y2;
            zlor.z2[cma[2]] = bufflor.z2;
			zlor.weight[cma[2]] = bufflor.weight;
            cma[2] += 1;
		}
	}
	if(cma[0] !=  nummainaxis[0] || cma[1] != nummainaxis[1] || cma[2] != nummainaxis[2]) cout<< "Something wrong with the number of lors for each main axis!!" <<endl;


	vector<lor>().swap(alllor);		//deallocate lor

	timeall.lorsorting += timing.GetCounter();

    // allocate device memory for storing LORs in each main axis.
	timinggpu.StartCounter();
    cudaMalloc((void**) &dev_xlor.x1, nummainaxis[0]*sizeof(float) );
	cudaMalloc((void**) &dev_xlor.y1, nummainaxis[0]*sizeof(float) ); 
	cudaMalloc((void**) &dev_xlor.z1, nummainaxis[0]*sizeof(float) ); 
	cudaMalloc((void**) &dev_xlor.x2, nummainaxis[0]*sizeof(float) ); 
	cudaMalloc((void**) &dev_xlor.y2, nummainaxis[0]*sizeof(float) ); 
	cudaMalloc((void**) &dev_xlor.z2, nummainaxis[0]*sizeof(float) ); 	
	cudaMalloc((void**) &dev_xlor.linevalue, nummainaxis[0]*sizeof(float) );
	cudaMalloc((void**) &dev_xlor.weight, nummainaxis[0]*sizeof(float) );

    cudaMalloc((void**) &dev_ylor.x1, nummainaxis[1]*sizeof(float) );
    cudaMalloc((void**) &dev_ylor.y1, nummainaxis[1]*sizeof(float) ); 
    cudaMalloc((void**) &dev_ylor.z1, nummainaxis[1]*sizeof(float) ); 
    cudaMalloc((void**) &dev_ylor.x2, nummainaxis[1]*sizeof(float) ); 
    cudaMalloc((void**) &dev_ylor.y2, nummainaxis[1]*sizeof(float) ); 
    cudaMalloc((void**) &dev_ylor.z2, nummainaxis[1]*sizeof(float) );
    cudaMalloc((void**) &dev_ylor.linevalue, nummainaxis[1]*sizeof(float) );
	cudaMalloc((void**) &dev_ylor.weight, nummainaxis[1]*sizeof(float) );	

    cudaMalloc((void**) &dev_zlor.x1, nummainaxis[2]*sizeof(float) );
    cudaMalloc((void**) &dev_zlor.y1, nummainaxis[2]*sizeof(float) ); 
    cudaMalloc((void**) &dev_zlor.z1, nummainaxis[2]*sizeof(float) ); 
    cudaMalloc((void**) &dev_zlor.x2, nummainaxis[2]*sizeof(float) ); 
    cudaMalloc((void**) &dev_zlor.y2, nummainaxis[2]*sizeof(float) ); 
    cudaMalloc((void**) &dev_zlor.z2, nummainaxis[2]*sizeof(float) ); 
	cudaMalloc((void**) &dev_zlor.linevalue, nummainaxis[2]*sizeof(float) );
	cudaMalloc((void**) &dev_zlor.weight, nummainaxis[2]*sizeof(float) );

    // stores LOR data into dev_xlor, dev_ylor, dev_zlor.
	cudaMemcpy( dev_xlor.x1, xlor.x1, nummainaxis[0]*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_xlor.y1, xlor.y1, nummainaxis[0]*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_xlor.z1, xlor.z1, nummainaxis[0]*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_xlor.x2, xlor.x2, nummainaxis[0]*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_xlor.y2, xlor.y2, nummainaxis[0]*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_xlor.z2, xlor.z2, nummainaxis[0]*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_xlor.weight, xlor.weight, nummainaxis[0]*sizeof(float), cudaMemcpyHostToDevice );

    cudaMemcpy( dev_ylor.x1, ylor.x1, nummainaxis[1]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_ylor.y1, ylor.y1, nummainaxis[1]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_ylor.z1, ylor.z1, nummainaxis[1]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_ylor.x2, ylor.x2, nummainaxis[1]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_ylor.y2, ylor.y2, nummainaxis[1]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_ylor.z2, ylor.z2, nummainaxis[1]*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_ylor.weight, ylor.weight, nummainaxis[1]*sizeof(float), cudaMemcpyHostToDevice );

    cudaMemcpy( dev_zlor.x1, zlor.x1, nummainaxis[2]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_zlor.y1, zlor.y1, nummainaxis[2]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_zlor.z1, zlor.z1, nummainaxis[2]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_zlor.x2, zlor.x2, nummainaxis[2]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_zlor.y2, zlor.y2, nummainaxis[2]*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_zlor.z2, zlor.z2, nummainaxis[2]*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_zlor.weight, zlor.weight, nummainaxis[2]*sizeof(float), cudaMemcpyHostToDevice );
	timeall.memoryIO += timinggpu.GetCounter();

    return 0;
}

//calculate snmatrix (target image in next iteration during image recon) based on projection and previous value. No normalization, no regularization, no blur
__global__ void calnewmatrix000(float *snmatrix, float *smatrix)
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
	int x = threadIdx.x, y = threadIdx.y, z = blockIdx.x;
	int jj;	//image index in 1D
	while(z < nz)
	{
		y = threadIdx.y;
		while(y < ny)
		{
			x = threadIdx.x;
			while(x < nx)
			{
			    jj = x + y * nx + z * nx * ny;
				snmatrix[jj] = snmatrix[jj] * smatrix[jj];
				x += blockDim.x;
			}
			y += blockDim.y;
		}
		z += gridDim.x;
	}
}

//Yes normalization, no regularization, no blur
__global__ void calnewmatrix100(float *snmatrix, float *smatrix, float *normimage)
{
    int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
    int x = threadIdx.x, y = threadIdx.y, z = blockIdx.x;
	int jj;
	while(z < nz)
	{
		y = threadIdx.y;
		while(y < ny)
		{
			x = threadIdx.x;
			while(x < nx)
			{
			    jj = x + y * nx + z * nx * ny;
       			snmatrix[jj] = snmatrix[jj] * smatrix[jj] / normimage[jj];
				x += blockDim.x;
			}
			y += blockDim.y;
		}
		z += gridDim.x;
	}
}       


//No normalization, yes regularization, no blur
__global__ void calnewmatrix010(float *snmatrix, float *smatrix, float *poimage)
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
    int x = threadIdx.x, y = threadIdx.y, z = blockIdx.x;
    int jj; //image index in 1D
	float beta = d_info[3], aa, bb, cc, laves = aves[0], lavep = avep[0];
    while(z < nz)
    {
        y = threadIdx.y;
        while(y < ny)
        {
            x = threadIdx.x;
            while(x < nx)
            {
                jj = x + y * nx + z * nx * ny;
				aa = 2.0f * beta / powf(laves,2);
				bb = 1.0f - 2.0f * beta * poimage[jj] / (laves * lavep);
				cc = -snmatrix[jj] * smatrix[jj];
                snmatrix[jj] = (-bb + sqrtf(powf(bb,2) - 4.0f * aa * cc)) / (2.0f * aa);
                x += blockDim.x;
            }
            y += blockDim.y;
        }
        z += gridDim.x;
    }

}

//Yes normalization, yes regularization, no blur
__global__ void calnewmatrix110(float *snmatrix, float *smatrix, float *normimage, float *poimage)
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
    int x = threadIdx.x, y = threadIdx.y, z = blockIdx.x;
    int jj; //image index in 1D
	float beta = d_info[3], aa, bb, cc, laves = aves[0], lavep = avep[0];
    while(z < nz)
    {
        y = threadIdx.y;
        while(y < ny)
        {
            x = threadIdx.x;
            while(x < nx)
            {
                jj = x + y * nx + z * nx * ny;
				aa = 2.0f * beta / powf(laves,2);
				bb = normimage[jj] - 2.0f * beta * poimage[jj] / (laves * lavep);
				cc = -snmatrix[jj] * smatrix[jj];
                snmatrix[jj] = (-bb + sqrtf(powf(bb,2) - 4.0f * aa * cc)) / (2.0f * aa);
                x += blockDim.x;
            }
            y += blockDim.y;
        }
        z += gridDim.x;
    }

}


//No normalization, yes regularization, yes blur.
template<typename Operation>
__global__ void calnewmatrix011(float *snmatrix, float *smatrix, float *poimage, float *bmatrix, float *allweight, float weight_scale, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff)
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
    int x = threadIdx.x, y = threadIdx.y, z = blockIdx.x;
    int jjj; //image index in 1D
	float beta = d_info[3], aa, bb, cc, laves = aves[0], lavep = avep[0], wi, a=d_info[0];
	int li,hi,lj,hj,lk,hk, idxy;
    while(z < nz)
    {
        y = threadIdx.y;
        while(y < ny)
        {
            x = threadIdx.x;
            while(x < nx)
            {
                jjj = x + y * nx + z * nx * ny;

				float scale=weight_scale;
				if(allweight != NULL) scale = allweight[jjj];

				li = max(0, x - d_indr[0]);
				lj = max(0, y - d_indr[1]);
				lk = max(0, z - d_indr[2]);
				hi = min(nx - 1, x + d_indr[0]);
				hj = min(ny - 1, y + d_indr[1]);
				hk = min(nz - 1, z + d_indr[2]);

				aa = 0.0f;
				bb = 0.0f;
		
				for(int ii= li; ii<= hi; ii++)
				{
					for(int jj = lj; jj<= hj; jj++)
					{
						for(int kk = lk; kk<= hk; kk++)
						{
							idxy = ii + jj * nx + kk * nx * ny;
							wi = Operation().blur(ii,jj,kk,x,y,z,a,psfimage,blurpara,numpara,numcoeff);	// Here x,y,z is center
                            wi = wi/scale;
							aa += wi;
                            bb += ((bmatrix[idxy] - smatrix[jjj] )/laves - poimage[idxy]/lavep) * wi;

		
						}
					}
				}

				aa = aa * 2.0f * beta / powf(laves,2);
				bb = bb * 2.0f * beta / laves + 1.0f;
				cc = -snmatrix[jjj] * smatrix[jjj];
                snmatrix[jjj] = (-bb + sqrtf(powf(bb,2) - 4.0f * aa * cc)) / (2.0f * aa);
                x += blockDim.x;
            }
            y += blockDim.y;
        }
        z += gridDim.x;
    }

}

//Yes normalization, yes regularization, yes blur
template<typename Operation>
__global__ void calnewmatrix111(float *snmatrix, float *smatrix, float *normimage, float *poimage, float *bmatrix, float *allweight, float weight_scale, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff)
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
    int x = threadIdx.x, y = threadIdx.y, z = blockIdx.x;
    int jjj; //image index in 1D
	float beta = d_info[3], aa, bb, cc, laves = aves[0], lavep = avep[0], wi, a = d_info[0];
	int li,hi,lj,hj,lk,hk, idxy;
    while(z < nz)
    {
        y = threadIdx.y;
        while(y < ny)
        {
            x = threadIdx.x;
            while(x < nx)
            {
                jjj = x + y * nx + z * nx * ny;
                
				float scale=weight_scale;
				if(allweight != NULL) scale = allweight[jjj];

				li = max(0, x - d_indr[0]);
				lj = max(0, y - d_indr[1]);
				lk = max(0, z - d_indr[2]);
				hi = min(nx - 1, x + d_indr[0]);
				hj = min(ny - 1, y + d_indr[1]);
				hk = min(nz - 1, z + d_indr[2]);

				aa = 0.0f;
				bb = 0.0f;
		
				for(int ii= li; ii<= hi; ii++)
				{
					for(int jj = lj; jj<= hj; jj++)
					{
						for(int kk = lk; kk<= hk; kk++)
						{
							idxy = ii + jj * nx + kk * nx * ny;
                            wi = Operation().blur(ii,jj,kk,x,y,z,a,psfimage,blurpara,numpara,numcoeff); // Here x,y,z is center
							wi = wi/scale;
							aa += wi;
                            bb += ((bmatrix[idxy] - smatrix[jjj])/laves - poimage[idxy]/lavep) * wi;
		
						}
					}
				}

				aa = aa * 2.0f * beta / powf(laves,2);
				bb = bb * 2.0f * beta / laves + normimage[jjj];
				cc = -snmatrix[jjj] * smatrix[jjj];
                snmatrix[jjj] = (-bb + sqrtf(powf(bb,2) - 4.0f * aa * cc)) / (2.0f * aa);
                x += blockDim.x;
            }
            y += blockDim.y;
        }
        z += gridDim.x;
    }

}

//calculate average of voxel values in an image
__global__ void calave(float *smatrix, float *gave)
{
	int msize = d_imageindex[3];
	int cacheindex = threadIdx.x, tid = threadIdx.x + blockIdx.x * blockDim.x;
   	__shared__ float buffave[reducsize];

	float buff = 0.0f;
	while(tid < msize)
	{
		buff += smatrix[tid];
		tid += blockDim.x * gridDim.x ;
	}
	buffave[cacheindex] = buff;
	__syncthreads();

	int i = blockDim.x / 2;
	while( i != 0)
	{
		if(cacheindex < i)  buffave[cacheindex] += buffave[cacheindex + i];
		__syncthreads();
		i /= 2;
	}

	if(cacheindex == 0) gave[blockIdx.x] = buffave[0];
}


// spatial-invariant Gaussian blurring
__device__ float Blur_Gaussian_Invariant::blur(int x, int y, int z, int ii, int jj, int kk, float a, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff) {
	return expf(-(powf(ii-x,2)/(2.0f * powf(d_bsgm[0],2)) + powf(jj-y,2)/(2.0f * powf(d_bsgm[1],2)) + powf(kk-z,2)/(2.0f * powf(d_bsgm[2],2))) * powf(a,2));
}

// blurring using a mask image
__device__ float Blur_Mask_Invariant::blur(int x, int y, int z, int ii, int jj, int kk, float a, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff) {
	int rx = x - ii + d_indr[0];
	int ry = y - jj + d_indr[1];
	int rz = z - kk + d_indr[2];
	return psfimage[rx + ry * (2 * d_indr[0] + 1) + rz * (2*d_indr[0]+1) * (2*d_indr[1]+1)];
}

// evaluate a quadratic function
__device__ float eval_quad(const float * __restrict__ blurpara, float r) {
	return blurpara[0] + blurpara[1] * r + blurpara[2] * r * r;
}

// spatial-variant Gaussian blurring
__device__ float Blur_Gaussian_Variant::blur(int x, int y, int z, int ii, int jj, int kk, float a, float *psfimage, const float * __restrict__ blurpara, const int numpara, const int numcoeff) {
	// ii,jj,kk is center for determining sigma
	int nx = d_imageindex[0], ny = d_imageindex[1];
	
	// coordinate of the voxel
	float ci = (ii + 0.5 ) * a - nx*a/2.0f;
	float cj = (jj + 0.5 ) * a - ny*a/2.0f;

	float r = ci * ci + cj * cj;
	r = sqrtf(r);
	
	float sint = cj / r, cost = ci / r;
	float di_rota = (x-ii)*a*cost + (y-jj)*a*sint;
	float dj_rota = (-1)*(x-ii)*a*sint + (y-jj)*a*cost;
	float dk_rota = (z-kk)*a;

	float para[5];	// if numpara != 5, need to change this.
	for(int i = 0; i < numpara; i++) para[i] = eval_quad(&blurpara[i*numcoeff], r);
	
	float er,et,ez;
	if((di_rota-para[4]) > 0) {
		float sigmao2 = para[1]* para[1];
		float di2 = (di_rota-para[4]) * (di_rota-para[4]);
		er = expf(-di2/(2.0f*sigmao2));
	}
	else {
		float sigmai2 = para[0]*para[0];
		float di2 = (di_rota-para[4]) * (di_rota-para[4]);
		er = expf(-di2/(2.0f*sigmai2));
	}

	float sigmat2 = para[2]*para[2];
	float dj2 = dj_rota*dj_rota;
	et = expf(-dj2/(2.0f*sigmat2));

	sigmat2 = para[3]*para[3];
	dj2 = dk_rota*dk_rota;
	ez = expf(-dj2/(2.0f*sigmat2));

	return er*et*ez;
}


// blur an image using spatial-invariant Gaussian blurring, spatial-variant Gaussian blurring, or a mask image.
template<typename Operation>
__global__ void gpublur(float *smatrix, float *bmatrix, float *allweight, float *psfimage, float weight_scale, const float * __restrict__ blurpara, const int numpara, const int numcoeff)
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
    int x = threadIdx.x, y = threadIdx.y, z = blockIdx.x;
    int i; //image index in 1D
    float sumval, a = d_info[0];
	int li,hi,lj,hj,lk,hk;
    float	wi;

    while(z < nz)
    {
        y = threadIdx.y;
        while(y < ny)
        {
            x = threadIdx.x;
            while(x < nx)
            {
                i = x + y * nx + z * nx * ny;
				li = max(0, x - d_indr[0]);
				lj = max(0, y - d_indr[1]);
				lk = max(0, z - d_indr[2]);
				hi = min(nx - 1, x + d_indr[0]);
				hj = min(ny - 1, y + d_indr[1]);
				hk = min(nz - 1, z + d_indr[2]);
		        sumval = 0.0f;
		
				for(int ii= li; ii<= hi; ii++)
				{
					for(int jj = lj; jj<= hj; jj++)
					{
						for(int kk = lk; kk<= hk; kk++)
						{
							wi = Operation().blur(x,y,z,ii,jj,kk,a,psfimage,blurpara,numpara,numcoeff);	
							int idxy = ii + jj * nx + kk * nx * ny;
							if(allweight != NULL) wi = wi / allweight[idxy]; 
							sumval += smatrix[idxy] * wi;
		
						}
					}
				}
				if(allweight == NULL) sumval = sumval / weight_scale;
				bmatrix[i] = sumval;
                x += blockDim.x;
            }
            y += blockDim.y;
        }
        z += gridDim.x;
    }


}

// For calculating a part of loglikelihood function value associated with LOR line value.
__global__ void calLogLike(float *xlinevalue, double *gloglike, const int lorindex)
{
	int cacheindex = threadIdx.x, tid = threadIdx.x + blockIdx.x * blockDim.x;
   	__shared__ double buffave[reducsize];

	double buff = 0.0;
	while(tid < lorindex)
	{
		if (xlinevalue[tid] > ThreshLineValue ) buff += logf(xlinevalue[tid]);
		tid += blockDim.x * gridDim.x ;
	}
	buffave[cacheindex] = buff;
	__syncthreads();

	int i = blockDim.x / 2;
	while( i != 0)
	{
		if(cacheindex < i)  buffave[cacheindex] += buffave[cacheindex + i];
		__syncthreads();
		i /= 2;
	}

	if(cacheindex == 0) gloglike[blockIdx.x] = buffave[0];
}

// For calculating a part of loglikelihood function value associated with the target image.
__global__ void calLogLikeS(float *smatrix, float *normimage, double *gloglike, const int msize, const int norma)
{
	int cacheindex = threadIdx.x, tid = threadIdx.x + blockIdx.x * blockDim.x;
   	__shared__ double buffave[reducsize];

	double buff = 0.0;
	while(tid < msize)
	{
		if(norma == 0) buff -= smatrix[tid];
		else {
			buff -= normimage[tid] * smatrix[tid];
		}
		tid += blockDim.x * gridDim.x ;
	}
	buffave[cacheindex] = buff;
	__syncthreads();

	int i = blockDim.x / 2;
	while( i != 0)
	{
		if(cacheindex < i)  buffave[cacheindex] += buffave[cacheindex + i];
		__syncthreads();
		i /= 2;
	}

	if(cacheindex == 0) gloglike[blockIdx.x] = buffave[0];
}

// For calculating the value of regularization term.
__global__ void calLogR(float *smatrix, float *poimage, double *gloglike, const int msize)
{
	int cacheindex = threadIdx.x, tid = threadIdx.x + blockIdx.x * blockDim.x;
   	__shared__ double buffave[reducsize];

	double buff = 0.0;
	while(tid < msize)
	{
		buff -= powf(smatrix[tid]/aves[0] - poimage[tid]/avep[0],2);
		tid += blockDim.x * gridDim.x ;
	}
	buffave[cacheindex] = buff;
	__syncthreads();

	int i = blockDim.x / 2;
	while( i != 0)
	{
		if(cacheindex < i)  buffave[cacheindex] += buffave[cacheindex + i];
		__syncthreads();
		i /= 2;
	}

	if(cacheindex == 0) gloglike[blockIdx.x] = buffave[0];
}

#endif

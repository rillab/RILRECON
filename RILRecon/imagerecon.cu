// Author: Hengquan Zhang
// This file contains source code for PET image reconstruction from coincidence data in binary format. 

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
#include <sys/time.h>
using namespace std;

#include "global_var.h"
#include "func.h"
#include "projection.h"
#include "timing.h"
#include "lor.h"
#include "macro.h"


//////////////////////////////////////
// Main function.
//////////////////////////////////////

int main(int argc, char* argv[])
{
	vector<string> vpara;
	string line;
	stringstream ss;
	ifstream config ("configRecon.txt");
    int itenum; //number of iterations.

    ///////////////////////////////////
    // read "configRecon.txt" file.
    ///////////////////////////////////

	if (config.is_open())
	{
		while ( getline (config,line) )
		{
			if(line.length() == 0) continue;
			
			vpara=explode(line,' ');
			if(vpara[0]=="FOV") {ss<<vpara[2];ss>>bndry[0];ss.clear();ss<<vpara[3];ss>>bndry[1];ss.clear();ss<<vpara[4];ss>>bndry[2];ss.clear();} else
			if(vpara[0]=="GridSize") {ss<<vpara[2];ss>>a;ss.clear();} else
			if(vpara[0]=="TorHalfWidth") {ss<<vpara[2];ss>>torhw;ss.clear();} else
			if(vpara[0]=="TorSigma") {ss<<vpara[2];ss>>torsgm;ss.clear();} else
			if(vpara[0]=="NumberOfIterations") {ss<<vpara[2];ss>>itenum;ss.clear();} else
			if(vpara[0]=="Regularization") {ss<<vpara[2];ss>>rgl;ss.clear();} else
			if(vpara[0]=="Normalization") {ss<<vpara[2];ss>>norma;ss.clear();} else
			if(vpara[0]=="ThreshNorm") {ss<<vpara[2];ss>>ThreshNorm;ss.clear();} else
			if(vpara[0]=="BetaR") {ss<<vpara[2];ss>>beta;ss.clear();} else
			if(vpara[0]=="BlurR") {ss<<vpara[2];ss>>blur;ss.clear();} else
			if(vpara[0]=="XsigmaRB") {ss<<vpara[2];ss>>bsgm[0];ss.clear();} else
			if(vpara[0]=="YsigmaRB") {ss<<vpara[2];ss>>bsgm[1];ss.clear();} else
			if(vpara[0]=="ZsigmaRB") {ss<<vpara[2];ss>>bsgm[2];ss.clear();} else
			if(vpara[0]=="ImagePSF") {ss<<vpara[2];ss>>imagepsf;ss.clear();} else
			if(vpara[0]=="PSFSigma") {
				ss<<vpara[2];ss>>psfsgm[0];ss.clear();
				ss<<vpara[3];ss>>psfsgm[1];ss.clear();
				ss<<vpara[4];ss>>psfsgm[2];ss.clear();} else
			if(vpara[0]=="MaskFile") {ss<<vpara[2];ss>>maskFileName;ss.clear();} else
			if(vpara[0]=="VoxelSize") {
				ss<<vpara[2];ss>>psfVoxelSize[0];ss.clear();
				ss<<vpara[3];ss>>psfVoxelSize[1];ss.clear();
				ss<<vpara[4];ss>>psfVoxelSize[2];ss.clear();} else
			if(vpara[0]=="BlurParaFile") {ss<<vpara[2];ss>>blurParaFile;ss.clear();} else
			if(vpara[0]=="BlurParaNum") {
				for(int i=0; i<2; i++) {ss<<vpara[2+i];ss>>blurParaNum[i];ss.clear();}
			} else
			if(vpara[0]=="RBCubeSize" && blur == 3) {
				for(int i=0; i<3; i++) {ss<<vpara[2+i];ss>>indr[i]; ss.clear();}	// Initialize indr here!!
			}


		}
		config.close();
	}
	else cout << "Unable to open config file"<<endl;



    ////////////////////////////////////////////////////////////////////////////
    // print the values read from "configRecon.txt" file for correctness check.
    ////////////////////////////////////////////////////////////////////////////

	cout<<"-------------------------------------------"<<endl;
	cout<<"Input parameters:"<<endl;
	cout<<"FOV: "<<bndry[0]<<" mm x "<<bndry[1]<<" mm x "<<bndry[2]<<" mm"<<endl;
	cout<<"Grid size: "<<a<<" mm"<<endl;
	cout<<"TOR half width: "<<torhw<<" mm"<<endl;
	cout<<"TOR sigma: "<<torsgm<<" mm"<<endl;
	cout<<"Number of iterations: "<<itenum<<endl;
	cout<<"Normalization?: "<<norma<<endl;
	if(norma != 0) cout<<"ThreshNorm: "<<ThreshNorm<<endl;
	cout<<"Regularization?: "<<rgl<<endl;
	cout<<"Use image-based PSF?: "<<imagepsf<<endl;
	
	float *blurpara, *dev_blurpara, *dev_allweight;	// Use when blur==3
	if(rgl==1)
	{
		cout<<"Beta for regularization: "<<beta<<endl;
		cout<<"Blur?: "<<blur<<endl;
        if(blur==1)
        {
            cout<<"Xsigma for blur: "<<bsgm[0]<<" mm"<<endl;
            cout<<"Ysigma for blur: "<<bsgm[1]<<" mm"<<endl;
            cout<<"Zsigma for blur: "<<bsgm[2]<<" mm"<<endl;
			for(int i=0; i<3; i++) {
				rads[i] = bsgm[i] * sqrt(-2. * log (bthresh)); 
				indr[i] = trunc(rads[i]/a);
				if(bsgm[i] == 0) bsgm[i] = 1.0;
			}

			weight_scale_2 = total_weight(indr, bsgm);
        }

		else if(blur==3)
		{
			cout<<"Blur parameter file name: "<<blurParaFile <<endl;
			cout<<"Blur parameter number: "<<blurParaNum[0] << " " << blurParaNum[1] <<endl;
			cout<<"Blur cube size: "<<indr[0] << " " << indr[1] << " " << indr[2] << endl;

			for(int i=0; i<3; i++) indr[i]=(indr[i]-1)/2;

			int tparasize = blurParaNum[0] * blurParaNum[1];

			readTxt<float> (blurParaFile, blurpara, tparasize);
			cout<<"Blur parameters: ";
			for(int i=0; i<tparasize; i++) cout<<blurpara[i]<<" ";
			cout<<endl;

			cudaMalloc((void**) &dev_blurpara, tparasize*sizeof(float) );
			cudaMemcpy(dev_blurpara, blurpara, tparasize * sizeof(float), cudaMemcpyHostToDevice);

		}
		
	}
	else beta = 0.;

	float *psfImage;    // Use when imagepsf==2
    if(imagepsf == 1) {
        cout<<"PSFSigma: "<<psfsgm[0]<<" "<<psfsgm[1]<<" "<<psfsgm[2]<<" mm"<<endl;
        for(int i=0; i<3; i++) {
			psfrads[i] = psfsgm[i] * sqrt(-2. * log (bthresh)); 
			psfindr[i] = trunc(psfrads[i]/a);
			if(psfsgm[i] == 0) psfsgm[i] = 1.0;
		}

		weight_scale_1 = total_weight(psfindr, psfsgm);
    }

	else if(imagepsf == 2) {
		cout<<"PSF File Name: "<<maskFileName << endl;
		cout << "PSF Image Voxel Size: " << psfVoxelSize[0] << " " << psfVoxelSize[1] << " " << psfVoxelSize[2] << endl;
        
		readBinary<float> (maskFileName,  psfImage, psfVoxelSize[0] * psfVoxelSize[1] * psfVoxelSize[2]);
		cout << "Finish reading PSF image." << endl;

		weight_scale_1 = total_weight_image(psfImage, psfVoxelSize[0] * psfVoxelSize[1] * psfVoxelSize[2]);

	
	}

	cout<<"-------------------------------------------"<<endl;


    ///////////////////////////////////////////
    // Initialization work.
    ///////////////////////////////////////////

    // calculate number of voxels in each axis
	msize = ceil(bndry[0] / a) * ceil(bndry[1] / a) *ceil( bndry[2] / a);
	int nx = ceil(bndry[0] / a);
	int ny = ceil(bndry[1] / a);
	int nz = ceil( bndry[2] / a);
	bndry[0] = nx * a; bndry[1] = ny * a; bndry[2] = nz * a;
	cout<<"FOV: "<<bndry[0]<<" mm x "<<bndry[1]<<" mm x "<<bndry[2]<<" mm"<<endl;
	cout << "Dimension of images: " << nx << " x " << ny << " x " << nz << endl;

    //copy fundamental variables to cuda constant memory.
	int *temp_imageindex;
	temp_imageindex = (int*) malloc(4 * sizeof(int));
	temp_imageindex[0] = nx;
	temp_imageindex[1] = ny;
	temp_imageindex[2] = nz;
	temp_imageindex[3] = msize;
	float *temp_info;
	temp_info = (float*) malloc(4 * sizeof(float));
	temp_info[0] = a;
	temp_info[1] = torhw;
	temp_info[2] = pow(torsgm,2);	
	temp_info[3] = beta;
	cudaMemcpyToSymbol(d_imageindex, temp_imageindex, 4 * sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_info, temp_info, 4 * sizeof(float), 0, cudaMemcpyHostToDevice);

    // define variables for image recon.
	float *dev_smatrix, *dev_snmatrix, *dev_poimage, *dev_bmatrix;
    float allave, *gave, *dev_gave, hostAvep;

	timinggpu.StartCounter();
	cudaMalloc((void**) &dev_smatrix, msize*sizeof(float) );
	cudaMalloc((void**) &dev_snmatrix, msize*sizeof(float) );
	smatrix = (float*) malloc(msize * sizeof(float));
	snmatrix = (float*) malloc(msize * sizeof(float));

	gave = (float*) malloc(blocksPerGrid * sizeof(float));
	cudaMalloc((void**) &dev_gave, blocksPerGrid*sizeof(float) );
    timeall.memoryIO += timinggpu.GetCounter();

	dim3 threads(threadsperBlock, threadsperBlock);
    nummainaxis = (int*) malloc(3 * sizeof(int));

	vector<double> logLikelihood, logR;	//value of objective functions in all iterations
	float *dev_blur_smatrix, *dev_blur_snmatrix;	// For image-based PSF
    if (imagepsf > 0) {
	    cudaMalloc((void**) &dev_blur_smatrix, msize*sizeof(float) );
	    cudaMalloc((void**) &dev_blur_snmatrix, msize*sizeof(float) );
    }

    if(rgl == 1) {
		string rglname=argv[3];
		readBinary<float> (rglname, poimage, msize);	// read prior image into memory
	    timinggpu.StartCounter();
		cudaMalloc((void**) &dev_poimage, msize*sizeof(float) );
		cudaMalloc((void**) &dev_bmatrix, msize*sizeof(float) );
		cudaMemcpy( dev_poimage, poimage, msize*sizeof(float), cudaMemcpyHostToDevice );
        timeall.memoryIO += timinggpu.GetCounter();

        //  calculate average value of voxels in the prior image.
		allave = 0.0;
		calave<<<blocksPerGrid, reducsize>>>(dev_poimage, dev_gave);
		cudaMemcpy(gave, dev_gave, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
		for(int jj=0; jj< blocksPerGrid; jj++)  allave += gave[jj];
		allave /= msize;
		cudaMemcpyToSymbol(avep, &allave, sizeof(float), 0, cudaMemcpyHostToDevice);
		cout<<"Prior image average value: "<<allave<<endl;
		hostAvep = allave;	//for storing A_P

        // Calculate dev_allweight        
        if (blur == 3) {
			cudaMalloc((void**) &dev_allweight, msize*sizeof(float));
			cudaMemset( dev_allweight, 0, msize*sizeof(float));
			cudaMemcpyToSymbol(d_indr, indr, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);

			int blockWidth = 8;
			dim3 dimBlock(blockWidth, blockWidth, blockWidth);
			dim3 dimGrid(ceil(nx*1.0f/blockWidth), ceil(ny*1.0f/blockWidth), ceil(nz*1.0f/blockWidth));
			total_weight_variant<<<dimGrid, dimBlock>>>(dev_allweight, dev_blurpara, blurParaNum[0], blurParaNum[1]);
		}

    }

    // normalization
    string filenorm=argv[4];	// name of the file that contains coincidence data for normalization
    get_normal_image(filenorm, nx, ny, nz);  // get values for normimage and dev_normimage if norma>0.


	// Initialize image-based psf model for forward and backward projection. Use when imagepsf==2.
	float *dev_psfImage;
	if(imagepsf == 2) {
		int totalSize = psfVoxelSize[0] * psfVoxelSize[1] * psfVoxelSize[2];
		cudaMalloc((void**) &dev_psfImage, totalSize*sizeof(float) );
		cudaMemcpy(dev_psfImage, psfImage, totalSize * sizeof(float), cudaMemcpyHostToDevice);
	}

	// With image based PSF, normimage needs blurring
	if(imagepsf > 0 && norma > 0) {
		float *dev_bnorm;
		cudaMalloc((void**) &dev_bnorm, msize*sizeof(float) );
		if (imagepsf == 1) blur_wrapper(dev_normimage, dev_bnorm, NULL, psfsgm, psfrads, psfindr, weight_scale_1);
		else if (imagepsf == 2) blur_wrapper(dev_normimage, dev_bnorm, NULL, psfVoxelSize, dev_psfImage, weight_scale_1);
		cudaMemcpy(normimage, dev_bnorm, msize*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(dev_bnorm);
	}

	// If use normalization, scale normimage here.
	if(norma > 0) {
		float maxnorm = 0.0;
		for(int i = 0; i < msize; i++) 
			if(maxnorm < normimage[i]) maxnorm = normimage[i];

		for(int i = 0; i < msize; i++) {
			float sen_temp = normimage[i] / maxnorm;
			if(sen_temp < ThreshNorm) sen_temp = ThreshNorm;
			normimage[i] = sen_temp;
		}

		cudaMemcpy(dev_normimage, normimage, msize*sizeof(float), cudaMemcpyHostToDevice);
			
	}


    // open file that contains input lors for image reconstruction. This should be after normalization file read.
	string filein=argv[1];
	cout<<"Sorting LORs and copying to device memory......"<<endl;
    preplor(filein); //  read lors in the file, sort lors, copy to cuda
	cout<<"Finish sorting and copying."<<endl;
	cudaMemcpyToSymbol(d_lorindex, nummainaxis, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
	cout<<"Number of LORs in each main axis (x,y,x): "<<nummainaxis[0]<<" "<<nummainaxis[1]<<" "<<nummainaxis[2]<<endl;


    //re-define beta, beta_new = beta * A / 2. Also initialize smatrix as an uniform image.
	if(norma == 0) {
		beta = beta * float(numline)/msize / 2.0f;
		for(int i=0; i<msize; i++) smatrix[i] = float(numline)/msize;
	}
	else {
		float sumNormimage = 0.0f;
		for(int i=0; i< msize; i++) sumNormimage += normimage[i];
		beta = beta * float(numline)/sumNormimage / 2.0f;
		for(int i=0; i<msize; i++) smatrix[i] = float(numline)/sumNormimage;
	}

	temp_info[3] = beta;
	cudaMemcpyToSymbol(d_info, temp_info, 4 * sizeof(float), 0, cudaMemcpyHostToDevice);
	

	//calculate the average voxel value in the target image. In this case, it remains unchanged across iterations.
	allave = smatrix[0];
	cudaMemcpyToSymbol(aves, &allave, sizeof(float), 0, cudaMemcpyHostToDevice);
	float hostAve = allave; //for storing the value of A.







    ///////////////////////////////////////////
    //start iterations for image reconstruction
    ///////////////////////////////////////////

	for(int ij=0; ij<itenum; ij++){

		double templogLikelihood = 0.0;
		double templogR = 0.0;

		cout<<"Starting "<<ij<<" iteration."<<endl;

		timinggpu.StartCounter();
		cudaMemcpy( dev_smatrix, smatrix, msize*sizeof(float), cudaMemcpyHostToDevice );
		cudaMemset( dev_snmatrix, 0, msize*sizeof(float));
		cudaMemset( dev_xlor.linevalue, 0, nummainaxis[0]*sizeof(float));
		cudaMemset( dev_ylor.linevalue, 0, nummainaxis[1]*sizeof(float)); 
		cudaMemset( dev_zlor.linevalue, 0, nummainaxis[2]*sizeof(float)); 
		timeall.memoryIO += timinggpu.GetCounter();

		// Forward and backward projection
		if(imagepsf == 1) {
			blur_wrapper(dev_smatrix, dev_blur_smatrix, NULL, psfsgm, psfrads, psfindr, weight_scale_1);
			proj_wrapper(dev_xlor, dev_ylor, dev_zlor, dev_blur_smatrix, dev_snmatrix);
			blur_wrapper(dev_snmatrix, dev_blur_snmatrix, NULL, psfsgm, psfrads, psfindr, weight_scale_1);
			cudaMemcpy( dev_snmatrix, dev_blur_snmatrix, msize*sizeof(float), cudaMemcpyDeviceToDevice );
		}
		else if(imagepsf == 2) {
			blur_wrapper(dev_smatrix, dev_blur_smatrix, NULL, psfVoxelSize, dev_psfImage, weight_scale_1);
			proj_wrapper(dev_xlor, dev_ylor, dev_zlor, dev_blur_smatrix, dev_snmatrix);
			blur_wrapper(dev_snmatrix, dev_blur_snmatrix, NULL, psfVoxelSize, dev_psfImage, weight_scale_1);
			cudaMemcpy( dev_snmatrix, dev_blur_snmatrix, msize*sizeof(float), cudaMemcpyDeviceToDevice );
		}


		else if(imagepsf == 0) proj_wrapper(dev_xlor, dev_ylor, dev_zlor, dev_smatrix, dev_snmatrix);
		else cout << "Unknown identifier for imagepsf!!" << endl;


		// Post processing
		if(rgl == 0)
		{
			if(norma == 0){
				timinggpu.StartCounter();
				calnewmatrix000<<<blocksPerGrid, threads>>>(dev_snmatrix, dev_smatrix);
				timeall.tpostimageprocess += timinggpu.GetCounter();
			}
			else{ 
				timinggpu.StartCounter();
				calnewmatrix100<<<blocksPerGrid, threads>>>(dev_snmatrix, dev_smatrix, dev_normimage);
				timeall.tpostimageprocess += timinggpu.GetCounter();
			}
		}
		
		else if(rgl == 1 && blur == 0)
		{

			if(norma == 0){
				timinggpu.StartCounter();
				calnewmatrix010<<<blocksPerGrid, threads>>>(dev_snmatrix, dev_smatrix, dev_poimage);
				timeall.tpostimageprocess += timinggpu.GetCounter();
			}
			else{ 
				timinggpu.StartCounter();
				calnewmatrix110<<<blocksPerGrid, threads>>>(dev_snmatrix, dev_smatrix, dev_normimage, dev_poimage);
				timeall.tpostimageprocess += timinggpu.GetCounter();
			}
		}

		else if(rgl == 1 && blur > 0)
		{
            //calculate the voxel values in blurred target image
			cudaMemset( dev_bmatrix, 0, msize*sizeof(float));
			if(blur == 1) blur_wrapper( dev_smatrix, dev_bmatrix, NULL, bsgm, rads, indr, weight_scale_2);
			else if(blur == 3) blur_wrapper(dev_smatrix, dev_bmatrix, dev_allweight, indr, dev_blurpara, blurParaNum[0], blurParaNum[1]) ;


			//calculate new image for this iteration
			if(norma == 0){
				timinggpu.StartCounter();
				if(blur==1) calnewmatrix011<Blur_Gaussian_Invariant><<<blocksPerGrid, threads>>>(dev_snmatrix, dev_smatrix, dev_poimage, dev_bmatrix, NULL, weight_scale_2, NULL, NULL, 0, 0);
				else if(blur==3) calnewmatrix011<Blur_Gaussian_Variant><<<blocksPerGrid, threads>>>(dev_snmatrix, dev_smatrix, dev_poimage, dev_bmatrix, dev_allweight, 0, NULL, dev_blurpara, blurParaNum[0], blurParaNum[1]);
				timeall.tpostimageprocess += timinggpu.GetCounter();
			}
			else{
				timinggpu.StartCounter();
				if(blur==1) calnewmatrix111<Blur_Gaussian_Invariant><<<blocksPerGrid, threads>>>(dev_snmatrix, dev_smatrix, dev_normimage, dev_poimage, dev_bmatrix, NULL, weight_scale_2, NULL, NULL, 0, 0);
				else if(blur==3) calnewmatrix111<Blur_Gaussian_Variant><<<blocksPerGrid, threads>>>(dev_snmatrix, dev_smatrix, dev_normimage, dev_poimage, dev_bmatrix, dev_allweight, 0, NULL, dev_blurpara, blurParaNum[0], blurParaNum[1]);
				timeall.tpostimageprocess += timinggpu.GetCounter();
			}
		}

		else cout<<"Unknown indentifier for regularization or blur!!!"<<endl;


		// Copy new image from device to host memory.
		timinggpu.StartCounter();
		cudaMemcpy(snmatrix, dev_snmatrix, msize*sizeof(float), cudaMemcpyDeviceToHost);
		timeall.memoryIO += timinggpu.GetCounter();

		cout<<"Finish "<<ij<<" iteration."<<endl;

		//write new image to file.
		ostringstream convert;
		convert<<(ij+1);
		string fileout=argv[2];
		fileout.append(convert.str());
		writeBinary<float> (fileout, snmatrix, msize);



		//calculate objective function values
		double *gloglike, *dev_gloglike;
		gloglike = (double*) malloc(blocksPerGrid * sizeof(double));
		cudaMalloc((void**) &dev_gloglike, blocksPerGrid*sizeof(double) );
		calLogLike<<<blocksPerGrid, reducsize>>>(dev_xlor.linevalue, dev_gloglike, nummainaxis[0]);
		cudaMemcpy(gloglike, dev_gloglike, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
		for(int iobj = 0; iobj < blocksPerGrid; iobj++) templogLikelihood += gloglike[iobj];
	
		calLogLike<<<blocksPerGrid, reducsize>>>(dev_ylor.linevalue, dev_gloglike, nummainaxis[1]);
		cudaMemcpy(gloglike, dev_gloglike, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
		for(int iobj = 0; iobj < blocksPerGrid; iobj++) templogLikelihood += gloglike[iobj];
	
		calLogLike<<<blocksPerGrid, reducsize>>>(dev_zlor.linevalue, dev_gloglike, nummainaxis[2]);
		cudaMemcpy(gloglike, dev_gloglike, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
		for(int iobj = 0; iobj < blocksPerGrid; iobj++) templogLikelihood += gloglike[iobj];


		calLogLikeS<<<blocksPerGrid, reducsize>>>(dev_smatrix, dev_normimage, dev_gloglike, msize, norma);
		cudaMemcpy(gloglike, dev_gloglike, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
		for(int iobj = 0; iobj < blocksPerGrid; iobj++) templogLikelihood += gloglike[iobj];


		if(rgl == 1 && blur == 0) calLogR<<<blocksPerGrid, reducsize>>>(dev_smatrix, dev_poimage, dev_gloglike, msize);
		else if(rgl == 1 && blur > 0) calLogR<<<blocksPerGrid, reducsize>>>(dev_bmatrix, dev_poimage, dev_gloglike, msize);
		if(rgl == 1) {
			cudaMemcpy(gloglike, dev_gloglike, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
			for(int iobj = 0; iobj < blocksPerGrid; iobj++) templogR += gloglike[iobj];	
		}


		templogR *= beta;
		
		logLikelihood.push_back(templogLikelihood);
		logR.push_back(templogR);



		//prepare for next iteration
		for(int iii=0; iii< msize; iii++)
		{
			smatrix[iii] = snmatrix[iii];
			snmatrix[iii] = 0.;
		}


	}


    /////////////////////////////////////////////////////
	// save the value of objective function to file.
    /////////////////////////////////////////////////////

	ofstream fObjFunc ("ObjectiveFuncValue.txt");
	if(fObjFunc.is_open()){
		for (int i=0; i< itenum; i++) fObjFunc << i << " "<< logLikelihood[i] << " " << logR[i] << " " << logLikelihood[i] + logR[i] << endl;
	}
	else cout<< "Can not open ObjectiveFuncValue.txt!!" <<endl;
	fObjFunc.close();

	timeall.printvalue();	//print out timing information about cuda execution.


    ///////////////////////////////////////////////////////////
	// Free dynamically allocated memory and GPU global memory.
    ///////////////////////////////////////////////////////////

	cudaFree(dev_xlor.x1); cudaFree(dev_xlor.y1); cudaFree(dev_xlor.z1); cudaFree(dev_xlor.x2); cudaFree(dev_xlor.y2); cudaFree(dev_xlor.z2);
    cudaFree(dev_ylor.x1); cudaFree(dev_ylor.y1); cudaFree(dev_ylor.z1); cudaFree(dev_ylor.x2); cudaFree(dev_ylor.y2); cudaFree(dev_ylor.z2);
    cudaFree(dev_zlor.x1); cudaFree(dev_zlor.y1); cudaFree(dev_zlor.z1); cudaFree(dev_zlor.x2); cudaFree(dev_zlor.y2); cudaFree(dev_zlor.z2);
	cudaFree(dev_xlor.linevalue); cudaFree(dev_ylor.linevalue); cudaFree(dev_zlor.linevalue);
	cudaFree(dev_smatrix);
	cudaFree(dev_snmatrix);
    free(nummainaxis);
    free(temp_imageindex);
    free(temp_info);
    free(xlor.x1); free(xlor.y1); free(xlor.z1); free(xlor.x2); free(xlor.y2); free(xlor.z2);
    free(ylor.x1); free(ylor.y1); free(ylor.z1); free(ylor.x2); free(ylor.y2); free(ylor.z2);
    free(zlor.x1); free(zlor.y1); free(zlor.z1); free(zlor.x2); free(zlor.y2); free(zlor.z2);	
	free(xlor.linevalue); free(ylor.linevalue); free(zlor.linevalue);
	free(smatrix);
	free(snmatrix);
    if(norma > 0) {free(normimage); cudaFree(dev_normimage);}
	if(imagepsf == 2) {free(psfImage); cudaFree(dev_psfImage);}
    if(imagepsf > 0) {cudaFree(dev_blur_smatrix); cudaFree(dev_blur_snmatrix);}
	if(rgl == 1 && blur == 3) {free(blurpara); cudaFree(dev_allweight); cudaFree(dev_blurpara);}
	if(rgl == 1) {free(poimage); cudaFree(dev_poimage); cudaFree(dev_bmatrix);}
	return 0;
}






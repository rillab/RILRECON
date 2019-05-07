#ifndef projection_H
#define projection_H

#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include "device_functions.h"

#include "lor.h"
#include "global_var.h"
#include "macro.h"

// forward projection of LORs with main axis being the x axis
__global__ void xfpro( cudalor lor, float *smatrix ) 
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
	float a = d_info[0], torhw = d_info[1], torsgm2 = d_info[2];
	int lornum = d_lorindex[0];

	__shared__ float cache[sharesize][sharesize];
	int tid ;
	int cacheIndex1 , cacheIndex2;

	float linevalue = 0.;
	float ulen2, t, oy, oz;
	int mlyy,mhyy,mlzz,mhzz;
	float x1,x2,y1,y2,z1,z2,weight;

	int tilenum1 = (ny + sharesize - 1)/ sharesize, tilenum2 = (nz + sharesize - 1) / sharesize ;

    // for each slice of the 3D image
	for(int i=0; i< nx; i++)  
	{

        // each slice is made up of a 2D array of tiles. This is due to limited GPU shared memory which cannot store the entire slice.
        for(int tn1 = 0; tn1 < tilenum1; tn1++)	
        {
		for(int tn2 = 0; tn2 < tilenum2; tn2++)
		{

        //load the voxel values of a tile of image into shared memory.
        cacheIndex1 = threadIdx.x;
		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < ny )	
        {
		cacheIndex2 = threadIdx.y;
        while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < nz)
        {
            cache[cacheIndex1][cacheIndex2] = smatrix[i + ((sharesize * tn1) + cacheIndex1) * nx + ((sharesize * tn2) + cacheIndex2) * nx * ny];
            cacheIndex2 += blockDim.y;
        }
            cacheIndex1 += blockDim.x;
        }
        __syncthreads();

        // calculate the forward projection of the tile of image to each LOR using OD-RT geometrical projector.
		tid = threadIdx.x + threadIdx.y * blockDim.x +  blockIdx.x * blockDim.x * blockDim.y;
		while(tid < lornum)
		{
			x1 = lor.x1[tid];
			y1 = lor.y1[tid];
			z1 = lor.z1[tid];
			x2 = lor.x2[tid];
			y2 = lor.y2[tid];
			z2 = lor.z2[tid];
			weight = lor.weight[tid];
			linevalue = 0.;
	
			ulen2 = powf(x1-x2,2) + powf(y1-y2,2) + powf(z1-z2,2);

			t = ( i * a - x1) / (x2 - x1);
			
			oy = y1 + t * (y2 - y1);
			oz = z1 + t * (z2 - z1);
			
			mlyy = max((int)truncf((oy - (SRTWO * torhw ))/a)+1, 0);
			mhyy = min((int)truncf((oy + (SRTWO * torhw ))/a), ny - 1);
			mlzz = max((int)truncf((oz - (SRTWO * torhw ))/a)+1, 0);
			mhzz = min((int)truncf((oz + (SRTWO * torhw ))/a), nz - 1);

			mlyy = max(mlyy, sharesize * tn1);
			mhyy = min(mhyy, sharesize * (tn1 + 1)-1);
			mlzz = max(mlzz, sharesize * tn2);
			mhzz = min(mhzz, sharesize * (tn2 + 1)-1);
			
			for(int ky = mlyy; ky <= mhyy; ky++)
			{
			    for(int kz = mlzz; kz <= mhzz; kz++)
			    {
					float dy = oy - ky*a, dz = oz - kz*a;
					float inner = dy * (y1-y2) + dz * (z1 - z2);
					float dst2 = dy * dy + dz * dz - inner * inner / ulen2;
					float maxdst2 = torhw * torhw;
			        if(dst2 < maxdst2) linevalue += cache[ky -sharesize * tn1 ][ kz - sharesize * tn2] * expf(-dst2/(2.0f * torsgm2)) * weight;
			    }
			}
			lor.linevalue[tid] += linevalue;
			//finish one tile for one lor
			tid += blockDim.x * blockDim.y * gridDim.x;
		}
		//finish one tile for all lors

		__syncthreads();
		}
		}
		//finish all tiles in a slice
	}
	//finish all slices

}


// back projection of LORs with main axis being the x axis
__global__ void xbpro( cudalor lor, float *snmatrix ) 
{
	int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
	float a = d_info[0], torhw = d_info[1], torsgm2 = d_info[2];
	int lornum = d_lorindex[0];

	__shared__ float cache[sharesize][sharesize];
	int tid ;
	int cacheIndex1 , cacheIndex2;

	float linevalue, rlinevalue;
	float ulen2, t, oy, oz;
	int mlyy,mhyy,mlzz,mhzz;
	float x1,x2,y1,y2,z1,z2,weight;

	int tilenum1 = (ny + sharesize - 1)/ sharesize, tilenum2 = (nz + sharesize - 1) / sharesize ;

    // for each slice of the 3D image
	for(int i=0; i< nx; i++)
	{
        // each slice is made up of a 2D array of tiles. This is due to limited GPU shared memory which cannot store the entire slice.
        for(int tn1 = 0; tn1 < tilenum1; tn1++)	
        {
		for(int tn2 = 0; tn2 < tilenum2; tn2++)
		{

        //initialize the shared memory storing the voxel values of a tile of image. The voxel values are initialized to be zero.
        cacheIndex1 = threadIdx.x;
		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < ny )
        {
		cacheIndex2 = threadIdx.y;
        while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < nz)
        {
            cache[cacheIndex1][cacheIndex2] = 0.0f;
            cacheIndex2 += blockDim.y;
        }
            cacheIndex1 += blockDim.x;
        }
        __syncthreads();

        // calculate the back projection of each LOR to the tile of image using OD-RT geometrical projector.
		tid = threadIdx.x + threadIdx.y * blockDim.x +  blockIdx.x * blockDim.x * blockDim.y;
		while(tid < lornum)
		{
			x1 = lor.x1[tid];
			y1 = lor.y1[tid];
			z1 = lor.z1[tid];
			x2 = lor.x2[tid];
			y2 = lor.y2[tid];
			z2 = lor.z2[tid];
			weight = lor.weight[tid];
	        linevalue = lor.linevalue[tid];
	
	        if(linevalue < ThreshLineValue) rlinevalue = 0.0f;
	        else rlinevalue = 1.0f / linevalue;
	
			ulen2 = powf(x1-x2,2) + powf(y1-y2,2) + powf(z1-z2,2);

			t = ( i * a - x1) / (x2 - x1);
			
			oy = y1 + t * (y2 - y1);
			oz = z1 + t * (z2 - z1);
			
			mlyy = max((int)truncf((oy - (SRTWO * torhw ))/a)+1, 0);
			mhyy = min((int)truncf((oy + (SRTWO * torhw ))/a), ny - 1);
			mlzz = max((int)truncf((oz - (SRTWO * torhw ))/a)+1, 0);
			mhzz = min((int)truncf((oz + (SRTWO * torhw ))/a), nz - 1);

			mlyy = max(mlyy, sharesize * tn1);
			mhyy = min(mhyy, sharesize * (tn1 + 1)-1);
			mlzz = max(mlzz, sharesize * tn2);
			mhzz = min(mhzz, sharesize * (tn2 + 1)-1);
			
			for(int ky = mlyy; ky <= mhyy; ky++)
			{
			    for(int kz = mlzz; kz <= mhzz; kz++)
			    {
					float dy = oy - ky*a, dz = oz - kz*a;
					float inner = dy * (y1-y2) + dz * (z1 - z2);
					float dst2 = dy * dy + dz * dz - inner * inner / ulen2;
					float maxdst2 = torhw * torhw;
			        if(dst2 < maxdst2) atomicAdd(&cache[ky -sharesize * tn1 ][ kz - sharesize * tn2], expf(-dst2/(2.0f * torsgm2)) * rlinevalue * weight) ;
			    }
			}

			//finish one lor for one tile
			tid += blockDim.x * blockDim.y * gridDim.x;
		}
		//finish all lors for one tile
		__syncthreads();

		//write the tile of image to global memory
		cacheIndex1 = threadIdx.x;

		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < ny )
        {
        cacheIndex2 = threadIdx.y;
        while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < nz)
        {
            atomicAdd(&snmatrix[i  + ((sharesize * tn1) + cacheIndex1) * nx + ((sharesize * tn2) + cacheIndex2) * nx * ny], cache[cacheIndex1][cacheIndex2]);
            cacheIndex2 += blockDim.y;
        }
            cacheIndex1 += blockDim.x;
        }
		__syncthreads();

		}
		}
		//finish all tiles in a slice
	}
	//finish all slices

}


// forward projection of LORs with main axis being the y axis. This is similar to xfpro.
__global__ void yfpro( cudalor lor, float *smatrix ) 
{

    int nx = d_imageindex[0], ny =d_imageindex[1], nz = d_imageindex[2];
    float a = d_info[0], torhw = d_info[1], torsgm2 = d_info[2];
	int lornum = d_lorindex[1];

	__shared__ float cache[sharesize][sharesize];
	int tid;
	int cacheIndex1, cacheIndex2;

    float linevalue = 0.;
    float ulen2, t, ox,oz;
    int mlxx,mhxx,mlzz,mhzz;
    float x1,x2,y1,y2,z1,z2,weight;

	int tilenum1 = (nx + sharesize - 1)/ sharesize, tilenum2 = (nz + sharesize - 1) / sharesize ;


    for(int i=0; i< ny; i++)
    {

        for(int tn1 = 0; tn1 < tilenum1; tn1++)	
        {
		for(int tn2 = 0; tn2 < tilenum2; tn2++)
		{

        cacheIndex1 = threadIdx.x;

		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < nx )	
        {
		cacheIndex2 = threadIdx.y;
        while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < nz)
        {

            cache[cacheIndex1][cacheIndex2] = smatrix[((sharesize * tn1) + cacheIndex1) + i * nx + ((sharesize * tn2) + cacheIndex2)* nx * ny];
			cacheIndex2 += blockDim.y;

        }
            cacheIndex1 += blockDim.x;
        }
        __syncthreads();

		tid = threadIdx.x + threadIdx.y * blockDim.x +  blockIdx.x * blockDim.x * blockDim.y;

	    while(tid < lornum)
	    {
	        x1 = lor.x1[tid];
	        y1 = lor.y1[tid];
	        z1 = lor.z1[tid];
	        x2 = lor.x2[tid];
	        y2 = lor.y2[tid];
	        z2 = lor.z2[tid];
			weight = lor.weight[tid];
			linevalue = 0.;
	
	        ulen2 = powf(x1-x2,2) + powf(y1-y2,2) + powf(z1-z2,2);


            t = ( i * a - y1) / (y2 - y1);

            ox = x1 + t * (x2 - x1);
            oz = z1 + t * (z2 - z1);

			mlxx = max((int)truncf((ox - (SRTWO * torhw ))/a)+1, 0);
			mhxx = min((int)truncf((ox + (SRTWO * torhw ))/a), nx - 1);
			mlzz = max((int)truncf((oz - (SRTWO * torhw ))/a)+1, 0);
			mhzz = min((int)truncf((oz + (SRTWO * torhw ))/a), nz - 1);

			mlxx = max(mlxx, sharesize * tn1);
			mhxx = min(mhxx, sharesize * (tn1 + 1)-1);
			mlzz = max(mlzz, sharesize * tn2);
			mhzz = min(mhzz, sharesize * (tn2 + 1)-1);
			                                                                                                                                                                          
            for(int kx = mlxx; kx <= mhxx; kx++)
            {
                for(int kz = mlzz; kz <= mhzz; kz++)
                {
					float dx = ox - kx*a, dz = oz - kz*a;
					float inner = dx * (x1-x2) + dz * (z1 - z2);
					float dst2 = dx * dx + dz * dz - inner * inner / ulen2;
					float maxdst2 = torhw * torhw;
					if(dst2 < maxdst2) linevalue += cache[kx -sharesize * tn1 ][ kz - sharesize * tn2] * expf(-dst2/(2.0f * torsgm2)) * weight;
				}
            }

    		lor.linevalue[tid] += linevalue;
	        tid += blockDim.x * blockDim.y * gridDim.x;
	    }
		__syncthreads();

        }
        }
    }


}


// back projection of LORs with main axis being the y axis. This is similar to xbpro.
__global__ void ybpro( cudalor lor, float *snmatrix )
{
    int nx = d_imageindex[0], ny =d_imageindex[1], nz = d_imageindex[2];
    float a = d_info[0], torhw = d_info[1], torsgm2 = d_info[2];
	int lornum = d_lorindex[1];

	__shared__ float cache[sharesize][sharesize];
	int tid ;
	int cacheIndex1 , cacheIndex2 ;

    float linevalue , rlinevalue;
    float ulen2, t, ox,oz;
    int mlxx,mhxx,mlzz,mhzz;
    float x1,x2,y1,y2,z1,z2,weight;

	int tilenum1 = (nx + sharesize - 1)/ sharesize, tilenum2 = (nz + sharesize - 1) / sharesize ;


    for(int i=0; i< ny; i++)
    {
        for(int tn1 = 0; tn1 < tilenum1; tn1++)	
        {
		for(int tn2 = 0; tn2 < tilenum2; tn2++)
		{

        cacheIndex1 = threadIdx.x;
		
		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < nx )	
        {
		cacheIndex2 = threadIdx.y;
        while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < nz)
        {
            cache[cacheIndex1][cacheIndex2] = 0.0f ;
            cacheIndex2 += blockDim.y;
        }
            cacheIndex1 += blockDim.x;
        }
        __syncthreads();

		tid = threadIdx.x + threadIdx.y * blockDim.x +  blockIdx.x * blockDim.x * blockDim.y;

	    while(tid < lornum)
	    {
	        linevalue = lor.linevalue[tid];
	
	        if(linevalue < ThreshLineValue) rlinevalue = 0.0f;
	        else rlinevalue = 1.0f / linevalue;
	
	        x1 = lor.x1[tid];
	        y1 = lor.y1[tid];
	        z1 = lor.z1[tid];
	        x2 = lor.x2[tid];
	        y2 = lor.y2[tid];
	        z2 = lor.z2[tid];
			weight = lor.weight[tid];
	
	        ulen2 = powf(x1-x2,2) + powf(y1-y2,2) + powf(z1-z2,2);
	
            t = ( i * a - y1) / (y2 - y1);

            ox = x1 + t * (x2 - x1);
            oz = z1 + t * (z2 - z1);

			mlxx = max((int)truncf((ox - (SRTWO * torhw ))/a)+1, 0);
			mhxx = min((int)truncf((ox + (SRTWO * torhw ))/a), nx - 1);
			mlzz = max((int)truncf((oz - (SRTWO * torhw ))/a)+1, 0);
			mhzz = min((int)truncf((oz + (SRTWO * torhw ))/a), nz - 1);

			mlxx = max(mlxx, sharesize * tn1);
			mhxx = min(mhxx, sharesize * (tn1 + 1)-1);
			mlzz = max(mlzz, sharesize * tn2);
			mhzz = min(mhzz, sharesize * (tn2 + 1)-1);

            for(int kx = mlxx; kx <= mhxx; kx++)
            {
                for(int kz = mlzz; kz <= mhzz; kz++)
                {
					float dx = ox - kx*a, dz = oz - kz*a;
					float inner = dx * (x1-x2) + dz * (z1 - z2);
					float dst2 = dx * dx + dz * dz - inner * inner / ulen2;
					float maxdst2 = torhw * torhw;
                    if(dst2 < maxdst2) atomicAdd(&cache[kx -sharesize * tn1 ][ kz - sharesize * tn2], expf(-dst2/(2.0f * torsgm2)) * rlinevalue * weight) ;

                }
            }
	    
			tid += blockDim.x * blockDim.y * gridDim.x;
		}

        __syncthreads();

		cacheIndex1 = threadIdx.x;
		
		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < nx )	
		{
		cacheIndex2 = threadIdx.y;
		while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < nz)
		{
		    atomicAdd(&snmatrix[((sharesize * tn1)  + cacheIndex1)  + i * nx + ((sharesize * tn2) + cacheIndex2) * nx * ny ],cache[cacheIndex1][cacheIndex2]);
		    cacheIndex2 += blockDim.y;
		}
		    cacheIndex1 += blockDim.x;
		}
		__syncthreads();

        }
        }
    }


}


// forward projection of LORs with main axis being the z axis. This is similar to xfpro.
__global__ void zfpro( cudalor lor, float *smatrix)
{
    int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
    float a = d_info[0], torhw = d_info[1], torsgm2 = d_info[2];
	int lornum = d_lorindex[2];

	__shared__ float cache[sharesize][sharesize];
	int tid ;
	int cacheIndex1 , cacheIndex2 ;

    float linevalue = 0.;
    float ulen2, t, ox, oy;
    int mlxx,mhxx,mlyy,mhyy;
    float x1,x2,y1,y2,z1,z2,weight;

	int tilenum1 = (nx + sharesize - 1)/ sharesize, tilenum2 = (ny + sharesize - 1) / sharesize ;

    for(int i=0; i< nz; i++)
    {

        for(int tn1 = 0; tn1 < tilenum1; tn1++)	
        {
		for(int tn2 = 0; tn2 < tilenum2; tn2++)
		{

        cacheIndex1 = threadIdx.x;

		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < nx )	
        {
		cacheIndex2 = threadIdx.y;
        while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < ny)
        {
            cache[cacheIndex1][cacheIndex2] = smatrix[((sharesize * tn1) + cacheIndex1) + ((sharesize * tn2) + cacheIndex2) * nx + i * nx * ny];
            cacheIndex2 += blockDim.y;
        }
            cacheIndex1 += blockDim.x;
        }
      	__syncthreads();

		tid = threadIdx.x + threadIdx.y * blockDim.x +  blockIdx.x * blockDim.x * blockDim.y;

	    while(tid < lornum)
	    {
	        x1 = lor.x1[tid];
	        y1 = lor.y1[tid];
	        z1 = lor.z1[tid];
	        x2 = lor.x2[tid];
	        y2 = lor.y2[tid];
	        z2 = lor.z2[tid];
			weight = lor.weight[tid];
			linevalue = 0.;
	
	        ulen2 = powf(x1-x2,2) + powf(y1-y2,2) + powf(z1-z2,2);

            t = ( i * a - z1) / (z2 - z1);

            oy = y1 + t * (y2 - y1);
            ox = x1 + t * (x2 - x1);

            mlyy = max((int)truncf((oy - (SRTWO * torhw ))/a)+1, 0);
            mhyy = min((int)truncf((oy + (SRTWO * torhw ))/a), ny - 1);
            mlxx = max((int)truncf((ox - (SRTWO * torhw ))/a)+1, 0);
            mhxx = min((int)truncf((ox + (SRTWO * torhw ))/a), nx - 1);

			mlxx = max(mlxx, sharesize * tn1);
			mhxx = min(mhxx, sharesize * (tn1 + 1)-1);
			mlyy = max(mlyy, sharesize * tn2);
			mhyy = min(mhyy, sharesize * (tn2 + 1)-1);

            for(int kx = mlxx; kx <= mhxx; kx++)
            {
                for(int ky = mlyy; ky <= mhyy; ky++)
                {
					float dy = oy - ky*a, dx = ox - kx*a;
					float inner = dy * (y1-y2) + dx * (x1 - x2);
					float dst2 = dy * dy + dx * dx - inner * inner / ulen2;
					float maxdst2 = torhw * torhw;
                    if(dst2 < maxdst2) linevalue += cache[kx -sharesize * tn1 ][ ky - sharesize * tn2] * expf(-dst2/(2.0f * torsgm2)) * weight;
                }
            }
	        lor.linevalue[tid] += linevalue;
	       	tid += blockDim.x * blockDim.y * gridDim.x;
		}

        __syncthreads();
        }
        }
    }

  
}


// back projection of LORs with main axis being the z axis. This is similar to xbprof.
__global__ void zbpro( cudalor lor, float *snmatrix )
{
    int nx = d_imageindex[0], ny = d_imageindex[1], nz = d_imageindex[2];
    float a = d_info[0], torhw = d_info[1], torsgm2 = d_info[2];
	int lornum = d_lorindex[2];

	__shared__ float cache[sharesize][sharesize];
	int tid ;
	int cacheIndex1 , cacheIndex2 ;

    float linevalue, rlinevalue;
    float ulen2, t, ox, oy;
    int mlxx,mhxx,mlyy,mhyy;
    float x1,x2,y1,y2,z1,z2,weight;

	int tilenum1 = (nx + sharesize - 1)/ sharesize, tilenum2 = (ny + sharesize - 1) / sharesize ;

    for(int i=0; i< nz; i++)
    {

        for(int tn1 = 0; tn1 < tilenum1; tn1++)	
        {
		for(int tn2 = 0; tn2 < tilenum2; tn2++)
		{

        cacheIndex1 = threadIdx.x;

		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < nx )	
        {
		cacheIndex2 = threadIdx.y;
        while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < ny)
        {
            cache[cacheIndex1][cacheIndex2] = 0.0f;
            cacheIndex2 += blockDim.y;
        }
            cacheIndex1 += blockDim.x;
        }
      	__syncthreads();

		tid = threadIdx.x + threadIdx.y * blockDim.x +  blockIdx.x * blockDim.x * blockDim.y;

	    while(tid < lornum)
	    {
	        x1 = lor.x1[tid];
	        y1 = lor.y1[tid];
	        z1 = lor.z1[tid];
	        x2 = lor.x2[tid];
	        y2 = lor.y2[tid];
	        z2 = lor.z2[tid];
			weight = lor.weight[tid];
	        linevalue = lor.linevalue[tid];
	
	        if(linevalue < ThreshLineValue) rlinevalue = 0.0f;
	        else rlinevalue = 1.0f / linevalue;
	
	        ulen2 = powf(x1-x2,2) + powf(y1-y2,2) + powf(z1-z2,2);

            t = ( i * a - z1) / (z2 - z1);

            oy = y1 + t * (y2 - y1);
            ox = x1 + t * (x2 - x1);

            mlyy = max((int)truncf((oy - (SRTWO * torhw ))/a)+1, 0);
            mhyy = min((int)truncf((oy + (SRTWO * torhw ))/a), ny - 1);
            mlxx = max((int)truncf((ox - (SRTWO * torhw ))/a)+1, 0);
            mhxx = min((int)truncf((ox + (SRTWO * torhw ))/a), nx - 1);

			mlxx = max(mlxx, sharesize * tn1);
			mhxx = min(mhxx, sharesize * (tn1 + 1)-1);
			mlyy = max(mlyy, sharesize * tn2);
			mhyy = min(mhyy, sharesize * (tn2 + 1)-1);

            for(int kx = mlxx; kx <= mhxx; kx++)
            {
                for(int ky = mlyy; ky <= mhyy; ky++)
                {
					float dy = oy - ky*a, dx = ox - kx*a;
					float inner = dy * (y1-y2) + dx * (x1 - x2);
					float dst2 = dy * dy + dx * dx - inner * inner / ulen2;
					float maxdst2 = torhw * torhw;
                    if(dst2 < maxdst2) atomicAdd(&cache[kx -sharesize * tn1 ][ ky - sharesize * tn2], expf(-dst2/(2.0f * torsgm2)) * rlinevalue * weight) ;
                }
            }

	       	tid += blockDim.x * blockDim.y * gridDim.x;
		}

        __syncthreads();


		cacheIndex1 = threadIdx.x;
		
		while(cacheIndex1 < sharesize && ((sharesize * tn1) + cacheIndex1) < nx )	
		{
		cacheIndex2 = threadIdx.y;
		while(cacheIndex2 < sharesize && ((sharesize * tn2) + cacheIndex2) < ny)
		{
		    atomicAdd(&snmatrix[((sharesize * tn1) + cacheIndex1) + ((sharesize * tn2) + cacheIndex2) * nx + i * nx * ny], cache[cacheIndex1][cacheIndex2]);
		    cacheIndex2 += blockDim.y;
		}
		    cacheIndex1 += blockDim.x;
		}
		
		__syncthreads();

        }
        }
    }

  
}


#endif

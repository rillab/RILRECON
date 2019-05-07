// Author: Hengquan Zhang
// This file contains the source code for trilinear interpolation of an image.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <list>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <limits>
using namespace std;

int main(int argc, char* argv[])
{
	float bndry[3] = {200., 200., 216.};    //FOV x, y, z. Unit: mm.
	float a = 4.; //grid size for input image.
	float b = 1.; //grid size for output image.
    int outs = 0;

    cout<<"Input FOVx"<<endl;
    cin>>bndry[0];
    cout<<"Input FOVy"<<endl;
    cin>>bndry[1];
    cout<<"Input FOVz"<<endl;
    cin>>bndry[2];
    cout<<"Input voxel size for input image"<<endl;
    cin>>a;
    cout<<"Input voxel size for output image"<<endl;
    cin>>b;	


	// calculate the number of voxels in each dimension
	int msize = ceil(bndry[0] / a) * ceil(bndry[1] / a) *ceil( bndry[2] / a);
	int nx = ceil(bndry[0] / a);
	int ny = ceil(bndry[1] / a);
	int nz = ceil( bndry[2] / a);

	int mbsize = ceil(bndry[0] / b) * ceil(bndry[1] / b) *ceil( bndry[2] / b);
	int nbx = ceil(bndry[0] / b);
	int nby = ceil(bndry[1] / b);
	int nbz = ceil( bndry[2] / b);
	

	ifstream fin;
	ofstream fout;
	string filein=argv[1], fileout=argv[1];
	fileout.append(".normal");

	fin.open(filein.c_str(), ios::in | ios::binary);
	fout.open(fileout.c_str(), ios::out | ios::binary);

	float buff;
	vector<float> smatrix;
	vector<float> sbmatrix;

	// read input image to smatrix
	for(int i=0; i< msize; i++)
	{
		fin.read((char *)&buff,sizeof(buff));
		smatrix.push_back( buff);
	}


	float ox,oy,oz;
	int ci,cj,ck;
	int apos[3];
	int npos[3] = {nx, ny, nz};
	

	float f1, f2, f3, f4, f5, f6, f7;
	float ax1, ax2, ay1, ay2;
	int aposx1, aposx2, aposy1, aposy2;


	try{
	for(int i=0; i< mbsize; i++)
	{
		// calculate the center of current voxel for the output image
		ci = (i % (nbx*nby)) % nbx;
		cj = (i % (nbx*nby)) / nbx;
		ck = i / (nbx*nby);
		ox = -bndry[0]/2. + (0.5 + ci) * b;
		oy = -bndry[1]/2. + (0.5 + cj) * b;
		oz = -bndry[2]/2. + (0.5 + ck) * b;

		// calculate the voxel index for the input image
		apos[0] = trunc((ox - (-bndry[0]/2. + 0.5 * a)) / a);  //be careful, it can be negative
		apos[1] = trunc((oy - (-bndry[1]/2. + 0.5 * a)) / a);
		apos[2] = trunc((oz - (-bndry[2]/2. + 0.5 * a)) / a);

		//make sure corners of square for interpolation are all within FOV
		for(int j=0; j< 3; j++)
		{
			if(apos[j]< 0) apos[j] += 1;
			if(apos[j] > npos[j]-2) apos[j] -= 1;	
		}


		// trilinear interpolation
		f1 = (ox - (-bndry[0]/2. + (0.5 + apos[0]) * a))/a * smatrix[(apos[0]+1) + apos[1] * (nx) + apos[2] * nx * ny] + (-bndry[0]/2. + ((0.5 + apos[0] + 1) * a)-ox)/a * smatrix[apos[0] + apos[1] * (nx) + apos[2] * nx * ny];
		
		f2 = (ox - (-bndry[0]/2. + (0.5 + apos[0]) * a))/a * smatrix[(apos[0]+1) + (apos[1]+1) * (nx) + apos[2] * nx * ny] + (-bndry[0]/2. + ((0.5 + apos[0] + 1) * a)-ox)/a * smatrix[apos[0] + (apos[1]+1) * (nx) + apos[2] * nx * ny];

		f3 = (ox - (-bndry[0]/2. + (0.5 + apos[0]) * a))/a * smatrix[(apos[0]+1) + apos[1] * (nx) + (apos[2]+1) * nx * ny] + (-bndry[0]/2. + ((0.5 + apos[0] + 1) * a)-ox)/a * smatrix[apos[0] + apos[1] * (nx) + (apos[2]+1) * nx * ny];
	
		f4 = (ox - (-bndry[0]/2. + (0.5 + apos[0]) * a))/a * smatrix[(apos[0]+1) + (apos[1]+1) * (nx) + (apos[2]+1) * nx * ny] + (-bndry[0]/2. + ((0.5 + apos[0] + 1) * a)-ox)/a * smatrix[apos[0] + (apos[1]+1) * (nx) + (apos[2]+1) * nx * ny];
	
		f5 = (oy - (-bndry[1]/2. + (0.5 + apos[1]) * a))/a * f2 + (-bndry[1]/2. + ((0.5 + apos[1] + 1) * a)-oy) /a * f1;

		f6 = (oy - (-bndry[1]/2. + (0.5 + apos[1]) * a))/a * f4 + (-bndry[1]/2. + ((0.5 + apos[1] + 1) * a)-oy) /a * f3;

		f7 = (oz - (-bndry[2]/2. + (0.5 + apos[2]) * a))/a * f6 + (-bndry[2]/2. + ((0.5 + apos[2] + 1) * a)-oz) /a * f5;

		sbmatrix.push_back( f7);

	}
	}catch(exception& e){
		cout<<e.what()<<endl;
		cout<<"Current position of voxel: "<<endl;
		cout<<ox<<" "<<oy<<" "<<oz<<endl;
	}


	// write output image to file.
	for(int i=0; i<mbsize; i++)
	{
		fout.write( (char*)&sbmatrix[i], sizeof(float));
	}


	fin.close();
	fout.close();




}

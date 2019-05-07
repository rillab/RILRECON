#ifndef lor_H
#define lor_H



//////////////////////////////////////
// Declare structures for lor.
//////////////////////////////////////

struct lor
{
	float x1;
	float y1;
	float z1;
	float x2;
	float y2;
	float z2;
	int mainaxis;	//0 for x, 1 for y, 2 for z
	float weight;
};

struct cudalor
{
	float *x1;
	float *y1;
	float *z1;
	float *x2;
	float *y2;
	float *z2;
	float *linevalue;
	float *weight;
};


#endif

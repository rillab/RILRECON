#ifndef macro_H
#define macro_H


#define sharesize 110  // tile width in a single direction for image recon. During image recon, the image is divided into 2D tiles. Each 2D tile is put into GPU shared memory for efficient memory read/write. 
#define blocksPerGrid 30  // number of blocks per grid for CUDA.
#define threadsperBlock 32  // number of threads per block for CUDA. The block used here is a 2D block with size being (threadsperBlock, threadsperBlock).
#define ThreshLineValue 0.001  // threshold for linevalue for image recon.
#define SRTWO 1.414  // value of sqrt(2)
#define reducsize 1024  // number of threads per block for CUDA. The block used here is a 1D block with size being reducsize.



// GPU error check.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


#endif

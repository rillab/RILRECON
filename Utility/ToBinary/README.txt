This computer program is for converting a txt file that contains coincidence events to binary format. The image reconstruction program use binary format of coincidence data as input for higher I/O efficiency.

To compile the code, simply type "make" in a terminal. Compiler: g++ 4.7.3.

The command for executing the code is as follows:
    ./convToBinary inputFile
The input txt file contains coincidence data. Each line has 6 numbers representing the coordinates of two photon interactions. A binary file named "inputFile.binary" will be output.

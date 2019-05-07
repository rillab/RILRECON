#ifndef timing_H
#define timing_H

#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "device_functions.h"

//////////////////////////////////////
// Class declaration for timer.
//////////////////////////////////////

// CPU timer.
class TimingCPU {

    private:
        long cur_time_;

    public:

        TimingCPU(): cur_time_(0) {};

        ~TimingCPU() {};

        void StartCounter(){
			struct timeval time;
			if(gettimeofday( &time, 0 )) return;
			cur_time_ = 1000000 * time.tv_sec + time.tv_usec;
		}

        double GetCounter(){
			struct timeval time;
			if(gettimeofday( &time, 0 )) return -1;

			long cur_time = 1000000 * time.tv_sec + time.tv_usec;
			double sec = (cur_time - cur_time_) / 1000000.0;
			if(sec < 0) sec += 86400;
		    cur_time_ = cur_time;
		
		    return 1000.*sec; //wall clock time (ms)
		}

}; 

struct PrivateTimingGPU {
    cudaEvent_t     start;
    cudaEvent_t     stop;
};

// GPU timer.
class TimingGPU {
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:

        TimingGPU() { privateTimingGPU = new PrivateTimingGPU; }

        ~TimingGPU() {}

        void StartCounter() {
            cudaEventCreate(&((*privateTimingGPU).start));
            cudaEventCreate(&((*privateTimingGPU).stop));
            cudaEventRecord((*privateTimingGPU).start,0);
        }

        float GetCounter(){
            float   time;
            cudaEventRecord((*privateTimingGPU).stop, 0);
            cudaEventSynchronize((*privateTimingGPU).stop);
            cudaEventElapsedTime(&time,(*privateTimingGPU).start,(*privateTimingGPU).stop);
            return time;
        }

} ; 

// all timing variables. 
class timevar{
	public:
		float txforward;
		float txbackward;
		float tyforward;
		float tybackward;
		float tzforward;
		float tzbackward;
		float tpostimageprocess;
		float memoryIO;
		float lorsorting;
		timevar(): txforward(0.0), txbackward(0.0), tyforward(0.0), tybackward(0.0), tzforward(0.0), tzbackward(0.0), tpostimageprocess(0.0), memoryIO(0.0), lorsorting(0.0) {}
		~timevar() {}
	
		void printvalue(){
			printf("%12s%12s%12s%12s%12s%12s%20s%12s%12s", "(ms) xf","yf", "zf","xb", "yb","zb","postprocess","memoryIO","lorsorting\n");
			printf("%12.1f%12.1f%12.1f%12.1f%12.1f%12.1f%20.1f%12.1f%12.1f\n", txforward, tyforward, tzforward, txbackward, tybackward, tzbackward, tpostimageprocess, memoryIO, lorsorting);
	}
	
};

#endif

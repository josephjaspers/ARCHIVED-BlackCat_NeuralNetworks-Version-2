////
////#include "BC_NN_Functions.h"
////
//#include "Tensor.h"
//	__global__
//	void sigmoid(float* x, unsigned sz) {
//		for (int i = 0; i < sz; ++i) {
//			x[i] = 1 / (1+ pow(2.71828, -x[i]));
//		}
//	}
//	__global__
//	void sigmoid_deriv(float* x, unsigned sz)  {
//		for (int i = 0; i < sz; ++i) {
//			x[i] *= (1 - x[i]);
//		}
//	}
////
//	void sigmoid(Tensor<float, GPU>& x) { sigmoid<<<10, 128>>>(x.data(), x.size());}
//		void sigmoid_deriv(Tensor<float, GPU>& x)  { sigmoid_deriv<<<10, 128>>>(x.data(), x.size());}
//
////

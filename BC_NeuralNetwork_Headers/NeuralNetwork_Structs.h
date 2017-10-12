/*
 * NonLinearityFunction.h
 *
 *  Created on: Aug 10, 2017
 *      Author: joseph
 */

#include "BC_NeuralNetwork_Definitions.h"

#ifndef NONLINEARITYFUNCTION_H_
#define NONLINEARITYFUNCTION_H_

namespace nonLin {

	void sigmoid(Tensor<double, CPU>& x);
	void sigmoid_deriv(Tensor<double, CPU>& x);
	__global__ void sig_deriv(float* data, unsigned sz);
	__global__ void sig(float* x, unsigned sz);
	void sigmoid(Tensor<float, GPU>& x);
	void sigmoid_deriv(Tensor<float, GPU>& x);
}

//controls sigmoid/tanh -- will be update
struct nonLinearityFunction {
	///nonLinearity nonLin = sigmoid;
public:
	tensor operator()(tensor x) {
		nonLin::sigmoid(x);
		return x;
	}
	tensor d(tensor x) {
		nonLin::sigmoid_deriv(x);
		return x;
	}
};

//bpStorages - controls back propagation data
#include <unordered_map>
#include <mutex>
#include "unq_thread.h"
struct bpStorage {
	std::vector<tensor> bp_storages;

	void store(tensor data) {
		bp_storages.push_back(data);
	}
	tensor& last() {
		return bp_storages.back();
	}
	tensor poll_last() {
		tensor data = std::move(bp_storages.back());
		bp_storages.pop_back();

		return std::move(data);
	}
	void clear() {
		bp_storages.clear();
	}
	void clearAll() {
		bp_storages.clear();
	}

	const tensor& operator()() {
		return bp_storages.back();
	}
	int size() {
		return bp_storages.size();
	}
};

//gradient storages -- updates weights/gradients -- assists with multithreading

#include "Tensor.h"
struct gradientStorage {
	std::vector<unsigned> gs_shape;

	tensor gStorage;

	void initialize(std::vector<unsigned> shape) {
		gs_shape = shape;
	}
	void addGradients(const tensor& gradients) {
		if (gStorage.isInitialized()) {
			gStorage -= gradients;
		} else {
			gStorage = tensor(gs_shape);
			gStorage -= gradients;
		}
	}
	void updateGradients(tensor& weights, scalar& learningRate) {
		weights += gStorage & learningRate;
		gStorage.zeros();
	}

	void clear() {
		gStorage = 0;
	}

};

#endif /* NONLINEARITYFUNCTION_H_ */

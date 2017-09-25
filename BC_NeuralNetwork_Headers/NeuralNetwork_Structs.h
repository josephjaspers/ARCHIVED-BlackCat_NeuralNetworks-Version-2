/*
 * NonLinearityFunction.h
 *
 *  Created on: Aug 10, 2017
 *      Author: joseph
 */


#include "BC_NeuralNetwork_Definitions.h"

#ifndef NONLINEARITYFUNCTION_H_
#define NONLINEARITYFUNCTION_H_


//controls sigmoid/tanh -- will be update
struct nonLinearityFunction {
	nonLinearity nonLin = sigmoid;
public:
	tensor operator()(tensor x) {
		for (int i = 0 ; i < x.size(); ++i) {
			x(i)=  1 / (1 + pow(2.71828, -x(i)));
		}
		return x;
	}
	tensor d(tensor x) {
		for (int i = 0; i < x.size(); ++i) {
			x(i) *= (1 - x(i));
		}
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

	const tensor& operator() (){
		return bp_storages.back();
	}
	int size() {
		 return bp_storages.size();
	}
};

//gradient storages -- updates weights/gradients -- assists with multithreading

#include "Tensor.h"
#include <atomic>
struct gradientStorage {
	std::vector<unsigned> gs_shape;

	unq_thread<tensor> gStorage;

	void initialize(std::vector<unsigned> shape) {
		gs_shape = shape;
	}
	void addGradients(const tensor& gradients) {
		if(gStorage().isInitialized()) {
			gStorage() -= gradients;
		} else {
			gStorage() =  tensor(gs_shape);
			gStorage() -= gradients;
		}
	}
	void updateGradients(tensor& weights, scalar& learningRate) {
		weights += gStorage() & learningRate;
		gStorage().fill(0);
	}

	void clear() {
		gStorage() = 0;
	}
	void clearCache() {
		gStorage.clearCache();
	}


};

#endif /* NONLINEARITYFUNCTION_H_ */

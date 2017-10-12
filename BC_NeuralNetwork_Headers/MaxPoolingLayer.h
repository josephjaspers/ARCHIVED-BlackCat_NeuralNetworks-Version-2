#include "Layer.h"
#include "BC_NN_Functions.h"
#include "nonLinearityFunction.cu"

class MaxPoolingLayer : public Layer {

	unsigned img_rows;
	unsigned img_cols;
	unsigned img_depth;
	unsigned stride;

	tensor pool;
	max_ids indexes;

public:

	MaxPoolingLayer(unsigned img_rows, unsigned img_cols, unsigned img_depth, unsigned stride) {
		this->img_rows = img_rows;
		this->img_cols = img_cols;
		this->img_depth = img_depth;
		this->stride = stride;

		this->pool = tensor({ (unsigned) ceil(img_rows / stride), (unsigned) ceil(img_cols / stride), (unsigned) img_depth });
	}
	vec forwardPropagation(vec input) {
		tensor data = std::move(input);
		data.reshape({ img_rows, img_cols, img_depth });

		std::pair<tensor, Tensor<unsigned>> max_index_pair = nonLin::max_pooling(data, 2);
		indexes.push_back(max_index_pair.second);
		bpX.store(max_index_pair.first);

		return next ? next->forwardPropagation(max_index_pair.first) : max_index_pair.first;
	}
	vec forwardPropagation_express(vec x) {
		tensor data = std::move(x);
		data.reshape({ img_rows, img_cols, img_depth });

		std::pair<tensor, Tensor<unsigned>> max_index_pair = nonLin::max_pooling(data, 2);

		return next ? next->forwardPropagation(max_index_pair.first) : max_index_pair.first;

	}
	vec backwardPropagation(vec dy) {
		tensor delta = std::move(dy);
		delta.reshape({ img_rows / stride, img_cols / stride, img_depth });
		tensor dx = nonLin::max_filling(delta, indexes.back(), stride);

		return prev->backwardPropagation(dx);
	}
	vec backwardPropagation_ThroughTime(vec dy) {

		tensor delta = std::move(dy);
		delta.reshape({ img_rows / stride, img_cols / stride, img_depth });
		tensor dx = nonLin::max_filling(delta, indexes.back(), stride);

		return prev->backwardPropagation(dx);
	}
	//NeuralNetwork update-methods
	void clearBackPropagationStorage() {
		bpX.clear();
		bpY.clear();
		indexes.clear();
	}
	void clearGradientStorage() {
	}
	void updateGradients() {
	}

};

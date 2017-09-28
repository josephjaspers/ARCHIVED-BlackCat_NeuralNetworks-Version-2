
#include "Layer.h"

class MaxPoolingLayer : public Layer {

	unsigned img_rows;
	unsigned img_cols;
	unsigned img_depth;
	unsigned krnl_r;
	unsigned krnl_c;
	unsigned stride;

	tensor pool;
	tensor data;

	MaxPoolingLayer(unsigned img_rows, unsigned img_cols, unsigned img_depth, unsigned kernel_rows, unsigned kernel_cols, unsigned stride) {
		this->img_rows = img_rows;
		this->img_cols = img_cols;
		this->img_depth = img_depth;
		this->krnl_r = kernel_rows;
		this->krnl_c = kernel_cols;
		this->stride = stride;

		this->pool = tensor({ceil(img_rows/stride), ceil(img_cols/stride), img_depth});
	}
	vec forwardPropagation(vec input) {
		data = std::move(input.reshape({img_rows, img_cols,img_depth}));
		data.reshape({img_rows, img_cols, img_depth});

		for (unsigned f = 0; f< img_depth; ++f) {
			for (unsigned r = 0; r< img_rows; r+= stride) {
				for (unsigned c = 0; c < img_cols; c+= stride) {
					pool[f][r/stride](c/stride) = data({f, r, c},{kernel_rows, kernel_cols}).max();
				}
			}
		}

		return next ? next->forwardPropagation(pool) : pool;
	}
	vec forwardPropagation_express(vec x) = 0;
	vec backwardPropagation(vec dy) = 0;
	vec backwardPropagation_ThroughTime(vec dy) = 0;
		//NeuralNetwork update-methods
	void clearBackPropagationStorage() = 0;
	void clearGradientStorage() = 0;
	void updateGradients() = 0;


};

#ifndef BLACKCAT_ConvolutionalLayer_h
#define BLACKCAT_ConvolutionalLayer_h

#include "Layer.h"

class ConvolutionalLayer : public Layer {
public:
	unsigned img_r;			//numb rows of input signal
	unsigned img_c;			//numb cols of input signal
	unsigned img_d;			//depth of the input signal

	unsigned numb_krnls;	//number of filters or kernels
	unsigned krnl_r;		//number of rows of the kernels
	unsigned krnl_c;		//number of cols of the kernels
	unsigned krnl_d;		//depth of the kernels

	unsigned out_r;			//output rows
	unsigned out_c;			//output cols
	unsigned out_d;			//output depth

	unsigned r_pos;			//number of rows for correlation/convolution positions
	unsigned c_pos;			//number of cols for correlation/convolution positions

	tensor y;					//output
	tensor w;  					//filter
	tensor b;  					//bias
	tensor dx;
	tensor w_gradientStorage;
	tensor b_gradientStorage;

public:

	ConvolutionalLayer(unsigned img_rows, unsigned img_cols, unsigned depth, unsigned filt_rows, unsigned filt_cols, unsigned n_filters) {
		this->img_d = depth;
		this->img_r = img_rows;
		this->img_c = img_cols;

		this->krnl_d = depth;
		this->numb_krnls = n_filters;
		this->krnl_r = filt_rows;
		this->krnl_c = filt_cols;

		this->out_r = img_rows - filt_rows + 1;
		this->out_c = img_cols - filt_cols + 1;
		this->out_d = n_filters;

		r_pos = img_rows - krnl_r + 1;
		c_pos = img_cols - krnl_c + 1;

		w = tensor({krnl_r, krnl_c, img_d, numb_krnls});
		b = tensor({out_r, out_c, numb_krnls});

		w_gradientStorage = tensor({krnl_r, krnl_c, img_d, numb_krnls});
		b_gradientStorage = tensor({out_r, out_c, numb_krnls});

		w.randomize(-5, 5);
		b.randomize(-5, 5);

		y = tensor(out_r, out_c, numb_krnls);
		dx = tensor(img_r, img_c, img_d);

	}
	tensor relu(tensor t) {
		for (unsigned i = 0; i < t.size(); ++i) {
			if (t(i)() > 1) {
				t(i)() = 1;
			} else {
				if (t(i)() < 0) {
					t(i)() = 0;
				}
			}
		}
		return t;
	}

	virtual vec forwardPropagation(vec data) {
		tensor x = data;
		x.reshape( { img_r, img_c, img_d });
		y = relu(w.x_corr_stack(2, x) + b);

		bpX.store(std::move(x));
		return next ? next->forwardPropagation(vec(std::move(y))) : std::move(vec(std::move(y)));

	}
	virtual vec forwardPropagation_express(vec data) {

		tensor x = data;
		x.reshape( { img_r, img_c, img_d });
		y = relu(w.x_corr_stack(2, x) + b);
		return next ? next->forwardPropagation_express(vec(std::move(y))) : std::move(vec(std::move(y)));

	}
	virtual vec backwardPropagation(vec error) {
		tensor dy = error;
		dy.reshape( { out_r, out_c, numb_krnls });
		tensor x(bpX.poll_last());

		b_gradientStorage -= dy;
		for (unsigned i = 0; i < numb_krnls; ++i)
			w_gradientStorage[i] -= x.x_corr_FilterError(2, dy[i]);

		if (prev) {
			dx.zeros();
			//sum the error
			for (unsigned i = 0; i < numb_krnls; ++i) {
				dx += w[i].x_corr_SignalError(2, dy[i]);
			}

			//send it off into the world
			return prev->backwardPropagation(vec(dx));
		} else {
			return error;
		}
	}
	virtual vec backwardPropagation_ThroughTime(vec error) {
		tensor dy = error;
		dy.reshape( { out_r, out_c, numb_krnls });
		tensor x(bpX.poll_last());

		b_gradientStorage -= dy;
		for (unsigned i = 0; i < numb_krnls; ++i)
			w_gradientStorage[i] -= x.x_corr_FilterError(2, dy[i]);

		if (prev) {
			dx.zeros();
			//sum the error
			for (unsigned i = 0; i < numb_krnls; ++i) {
				dx += w[i].x_corr_SignalError(2, dy[i]);
			}

			//send it off into the world
			return prev->backwardPropagation(vec(dx));
		} else {
			return error;
		}	}

	//NeuralNetwork update-methods
	virtual void clearBackPropagationStorage() {
		bpX.clear();
	}
	virtual void clearGradientStorage() {
		w_gradientStorage.zeros();
		b_gradientStorage.zeros();
	}
	virtual void updateGradients() {
		w += w_gradientStorage & lr;
		b += b_gradientStorage & lr; // / (output_rows * output_cols * n_filters);
	//	w.print();
	}
};
#endif

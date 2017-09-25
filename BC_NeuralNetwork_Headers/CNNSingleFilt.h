#ifndef CNN_singleFilt_h
#define CNN_singleFilt_h

#include "Layer.h"

class CNN_singleFilt: public Layer {
public:
	unsigned img_r;			//numb rows of input signal
	unsigned img_c;			//numb cols of input signal
	unsigned img_d;			//depth of the input signal

	unsigned krnl_r;		//number of rows of the kernels
	unsigned krnl_c;		//number of cols of the kernels
	unsigned krnl_d;		//depth of the kernels

	unsigned out_r;			//output rows
	unsigned out_c;			//output cols

	unsigned r_pos;			//number of rows for correlation/convolution positions
	unsigned c_pos;			//number of cols for correlation/convolution positions

	tensor y;					//output
	tensor w;  					//filter
	tensor b;  					//bias
	tensor dx;
	tensor w_gradientStorage;
	tensor b_gradientStorage;

public:

	CNN_singleFilt(unsigned img_rows, unsigned img_cols, unsigned depth, unsigned filt_rows, unsigned filt_cols) {
		this->img_d = depth;
		this->img_r = img_rows;
		this->img_c = img_cols;

		this->krnl_d = depth;
		this->krnl_r = filt_rows;
		this->krnl_c = filt_cols;

		this->out_r = img_rows - filt_rows + 1;
		this->out_c = img_cols - filt_cols + 1;

		r_pos = img_rows - krnl_r + 1;
		c_pos = img_cols - krnl_c + 1;

		w = tensor(krnl_r, krnl_c, img_d);
		b = tensor(out_r, out_c);

		w_gradientStorage = tensor(krnl_r, krnl_c, img_d);
		b_gradientStorage = tensor(out_r, out_c);

		w.randomize(-1, 2);
		b.randomize(-1, 2);


		y = tensor(out_r, out_c);
		dx = tensor(img_r, img_c, img_d);


		//input dimesnions = IMG_R, IMG_C, IMG_D
		//out   dimensions = OUT_R, OUT_C, N_Filters
		//weightdimensions = KRN_R, KRN_C, IMG_D, N_FILTERS

		//

	}
	virtual vec forwardPropagation(vec data) {
		tensor x = data;
		x.reshape( { img_r, img_c, img_d });
		y = g(w.x_corr(2, x) + b);

		bpX.store(std::move(x));
		return next ?
				next->forwardPropagation(vec(std::move(y))) :
				std::move(vec(std::move(y)));

	}
	virtual vec forwardPropagation_express(vec data) {

		tensor x = data;
		x.reshape( { img_r, img_c, img_d });
		y = g(w.x_corr(2, x) + b);
return next ?
		next->forwardPropagation_express(vec(std::move(y))) :
		std::move(vec(std::move(y)));

	}
	virtual vec backwardPropagation(vec error) {
		//Shape the error to apropriate dimensions
		tensor dy = error;
		dy.reshape( { out_r, out_c });
		//get the last inputs
		tensor x(bpX.poll_last());
		//get bias gradient
		b_gradientStorage -= dy;

		//calculate weight gradients
				unsigned e_id = 0;
				for (unsigned f = 0; f < 1; ++f) {			//for each filter
					for (unsigned r = 0; r < r_pos; ++r) {	//multiply the subtensor by the value
						for (unsigned c = 0; c < c_pos; ++c) {
							w_gradientStorage -= x( { 0, r, c }, { krnl_r,krnl_c, img_d }) & dy(e_id);
							++e_id;
							e_id = e_id % dy.size();
						}
					}
				}

//		w_gradientStorage.print();
//		w.printDimensions();
//		dy.printDimensions();
//		x.printDimensions();

//		int wait;
//		std::cin >> wait;

		if (prev) {
//			std::cout << " HERE " << std::endl;
//			dx.printDimensions();
//			dy.printDimensions();
//			w.printDimensions();
			dx[0] = w[0].x_corr_full(2, dy);

			return prev->backwardPropagation(vec(dx));
			   static void cross_correlation_noAdjust(number_type* s, unsigned cor_mv, unsigned order, const  unsigned* store_ld, const number_type* filter,const  unsigned * f_ld,const  unsigned* f_ranks,

		} else {
			return error;
		}
	}
	virtual vec backwardPropagation_ThroughTime(vec dy) {
		return dy;
	}

	//NeuralNetwork update-methods
	virtual void clearBackPropagationStorage() {
		bpX.clear();
	}
	virtual void clearGradientStorage() {
		w_gradientStorage = 0;
		b_gradientStorage = 0;
	}
	virtual void updateGradients() {
		w += w_gradientStorage & lr;
		b += b_gradientStorage & lr; // / (output_rows * output_cols * n_filters);
	}
};
#endif


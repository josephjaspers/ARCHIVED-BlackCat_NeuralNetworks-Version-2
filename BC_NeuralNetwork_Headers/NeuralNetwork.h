
#ifndef BLACKCAT_neuralnet_h
#define  BLACKCAT_neuralnet_h

#include "Layer.h"

class NeuralNetwork : public Layer {

	Layer* last = nullptr;
	Layer* first = nullptr;
public:

	void add(Layer* l) {
		if (first == nullptr) {
			first = l;
			last = first;
		} else {
			last->next = l;
			l->prev = last;
			last = l;
		}
	}
	vec forwardPropagation( vec x) { return first->forwardPropagation(x);}
	vec forwardPropagation_express( vec x)  { return first->forwardPropagation_express(x);}
	vec backwardPropagation( vec x) { return last->backwardPropagation(x);}
	vec backwardPropagation_ThroughTime( vec x) { return last->backwardPropagation_ThroughTime(x);}

	 void clearBackPropagationStorage() {
		Layer* ref_first = first;
		while (ref_first != last->next) {
			ref_first->clearBackPropagationStorage();
			ref_first = ref_first->next;
		}
	}
	 void clearGradientStorage() {
		Layer* ref_first = first;
		while (ref_first != last->next) {
			ref_first->clearGradientStorage();
			ref_first = ref_first->next;
		}
	}
	 void updateGradients() {
		Layer* ref_first = first;
		while (ref_first != last->next) {
			ref_first->updateGradients();
			ref_first = ref_first->next;
		}
	}
	 void train(const std::vector<vec>& inputs, const std::vector<vec>& outputs, unsigned iters) {
		 while (iters > 0) {
			 train(inputs, outputs);
			 --iters;
		 }
	 }
	 void train(const std::vector<vec>& inputs, const std::vector<vec>& outputs) {

		 for (int i = 0; i < inputs.size(); ++i) {
			 vec hyp = forwardPropagation(inputs[i]);
			 backwardPropagation(hyp - outputs[i]);
			 update();
		 }

	 }
	 vec abs(vec v) {
		 for (unsigned i = 0; i < v.size(); ++i) {
			 if (v(i) < 0) {
				 v(i) *= -1;
			 }
		 }
		 return v;
	 }
	 double sum(vec v) {
		 double sum =0;
		 for (unsigned i = 0; i < v.size(); ++i) {
			 sum += v(i);
		 }
		 return sum;
	 }
	 int maxId(const vec& in) {

		 double max1 = in(0);
		 int index = 0;
		 for (unsigned i = 1; i < in.size(); ++i) {
			 if (max1 < in(i)) {
				 max1 = in(i);
				 index = i;
			 }
		 }
		 return index;
	 }

	 double test(const std::vector<vec>& inputs, const std::vector<vec>& outputs) {


		 double percentcorrect = 0;
		 vec error = vec(outputs[0].size());
		 error = 0;
		 for (int i = 0; i < inputs.size(); ++i) {
			 vec hyp = forwardPropagation_express(inputs[i]);
			 vec residual = hyp - outputs[i];
			 error += abs(residual);

			 if (maxId(hyp) == maxId(outputs[i])) {
				 percentcorrect += 1;
			 }
		 }
		 std::cout << " numb correct = " << percentcorrect << std::endl;
		 std::cout << " percent correct = " << percentcorrect / inputs.size() << std::endl;
		 return sum(error)/ inputs.size();
	 }
	 void update() {
			Layer* ref_first = first;
			while (ref_first != last->next) {
				ref_first->updateGradients();
				ref_first->clearGradientStorage();
				ref_first->clearBackPropagationStorage();

				ref_first = ref_first->next;
			}
	 }

};



#endif

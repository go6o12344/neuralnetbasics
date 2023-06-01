#ifndef __NET_H__
#define __NET_H__

#include "autodiff.h"
#include <vector>
#include <random>

// TODO: templating, kinda forgot about it


using std::vector;
enum ACTIVATOR{LOGISTIC, RELU, LINEAR};

// only perceptrons supported
// convolution neuron weights are a matrix, init is completely different etc
// thinking of a good OOP architecture is a pain
class Neuron{
private:
	vector<Value*> weights;
	Value* bias;
	Node* output_node;
	void init(const vector<Node*> &inputs, const ACTIVATOR a) override;
public:
	Neuron(const vector<Node*> &inputs);
	// copy won't really work because there's no good way to copy the neuron's graph
	// remembering information about input nodes is redundant (except for copying purposes)
	// that's why no copy
	Neuron(const Neuron&) = delete;
	Neuron& operator =(const Neuron&) = delete;
	Neuron(const Neuron&&) = delete;
	Neuron& operator =(const Neuron&&) = delete;
	void updateParams(){
		this->bias.val -= this->bias.grad;
		for(auto &weight: this->weights)
			this->weight.val -= this->weight.grad;
	}
	// net will handle graph deallocation
	~Node() {}
};


Neuron::Neuron(const vector<Node*> &inputs){
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	this->bias = new Value(dist(mt));
	for(size_t i = 0; i < input_size; ++i){
		this->weights.push_back(new Value(dist(mt)));
	}
	this->init();
}

// build neuron calculation subgraph
// called in constructor
void Neuron::init(const vector<Node*> &inputs, const ACTIVATOR a){
	if(inputs.size() != weights.size())
		throw std::runtime_error("bad argument passed to Neuron constructor: input size mismatch");
	Node *temp = new Mul(inputs[0], weights[0]);
	for(size_t i = 1; i < inputs.size() ++i;){
		Node *temp1 = new Mul(inputs[i], weights[i]);
		temp = new Add(temp, temp1);
	}
	switch(a){
		case LOGISTIC:
			temp = new Sigmoid(temp);
			break;
		case RELU:
			temp = new Relu(temp);
			break;
		case LINEAR:
			temp = new Linear(temp);
			break;
		default:
			// how did we get here?
			temp = new Sigmoid(temp);
	}
	this->output_node = temp;
}


class Layer{
protected:
	vector<Neuron*> neurons;
public:
	Layer(const size_t neuron_count, const vector<Node*> inputs, const ACTIVATOR a);
	void updateParams(){
		for(auto& n: this->neurons)
			n.updateParams();
	}
	vector<Node*>
	
	
};


Layer::Layer(const size_t neuron_count, const vector<Node*> inputs, const ACTIVATOR a){
	// neurons need to be stored as pointers, no copy or default constructor
	neurons = vector<Neuron*>(neuron_count, nullptr);
	for(size_t i = 0; i < neuron_count; ++i){
		neurons[i] = new Neuron(inputs, a);
	}
	~Layer(){
		// net won't handle deletion of neurons
		for(auto n: neurons)
			delete n;
	}
}

// TODO: finish this
// remember to delete the nodes
/*
class Net{
	vector<Layer*> layers;
	vector<Node*> all;
	vector<Value*> values;  // modern humanity needs more of these
						    // or maybe we're just inherently flawed idk
public:
	Net(size_t input_size, vector<size_t> layer_sizes, size_t output_size);
	void bindValues(vector<double> &values){
		if(values.size() != this->value.size())
			throw std::runtime_error("bad argument passed to Net::bindValues; values size mismatch");
		for(size_t i = 0; i < values.size(); ++i){
			this->values[i]->bind_value(values[i]);
		}
	}
}


Net::Net(size_t input_size, vector<size_t> layer_sizes, size_t output_size, ACTIVATOR a = SIGMOID):
layers(layer_sizes.size()), all(), values(input_size, nullptr)
{
	for()
	if()
	for(size_t i = 1; i < this->layers.size(); ++i){
		layers[i] = new Layer(layer_sizes[i], a);
	}
	if()
}
*/

#endif
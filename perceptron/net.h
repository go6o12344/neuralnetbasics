#ifndef __NET_H__
#define __NET_H__

#include "autodiff_micro.h"
#include <vector>

// delete everything but leaves
// leaves will only be weights, biases or variables
void handleMemory(Value* val){
	if(val->children.size() == 0){
		delete val;
		return;
	}
	for(auto child: val->children)
		handleMemory(child);
	delete val;
}


class Module{
public:
	virtual std::vector<Value*> getParams() = 0;
	void descent(){
		auto params = this->getParams();
		for(auto param: params)
			param->val -= param->grad;
	}
	void resetGrad(){
		auto params = this->getParams();
		for(auto param: params)
			param->grad = 0;
	}
};


class Neuron : public Module{
	// those never have children
	// activator functions break if it's a vector of instances, vector returns copies of values on read (?, pretty sure that's what docs said)
	std::vector<Value*> weights;
	Value* bias;
	Value* (*activator)(Value*);
	void copy(const Neuron &other);
	void del();
	void move(Neuron &&other);
public:
	Neuron(size_t weight_count, double bias, Value* (*activator)(Value*) = sigmoid):Module(), weights{weight_count, nullptr},  bias{new Value{bias}}, activator{activator}
	{
		for(size_t i = 0; i < weight_count; ++i){
			weights[i] = new Value(1.);
		}
	}
	Neuron(const Neuron &other);
	Neuron& operator =(const Neuron &other);
	Neuron(Neuron &&other);
	Neuron& operator =(Neuron &&other);
	~Neuron();
	Value* activate(std::vector<Value*> &args);
	
	// need those to do gradient descent
	// this will be used by the network to update the value objects on train
	std::vector<Value*> getParams(){
		std::vector<Value*> res{weights};
		res.push_back(bias);
		return res;
	}
	
};


void Neuron::copy(const Neuron &other){
	// bias has no children, so we don't care about issues with the copy constructor for now
	bias = new Value(*other.bias);
	weights = std::vector<Value*>(other.weights.size(), nullptr);
	// the above statement but for weights
	for(size_t i = 0; i < weights.size(); ++i)
		weights[i] = new Value(*other.weights[i]);
	activator = other.activator;
}


void Neuron::del(){
	for(auto w: weights)
		delete w;
	delete bias;
}


void Neuron::move(Neuron &&other){
	// bias has no children, so we don't care about issues with the copy constructor for now
	bias = other.bias;
	other.bias = nullptr;
	weights = std::vector<Value*>(other.weights.size(), nullptr);
	for(size_t i = 0; i < weights.size(); ++i){
		other.weights[i] = nullptr;
	}
	activator = other.activator;
}


Neuron::Neuron(const Neuron &other){
	copy(other);
}


Neuron& Neuron::operator =(const Neuron &other){
	if(this != &other){
		del();
		copy(other);
	}
	return *this;
}


Neuron::Neuron(Neuron &&other){
	move(std::move(other));
}


Neuron& Neuron::operator =(Neuron &&other){
	if(this != &other){
		del();
		move(std::move(other));
	}
	return *this;
}


Neuron::~Neuron(){
	del();
}


Value* Neuron::activate(std::vector<Value*> &args){
	if(args.size() == 0)
		return bias;
	if(args.size() != weights.size())
		throw std::runtime_error("wrong argument size for activation");
	Value *temp;
	temp = multiply(args[0], weights[0]);
	for(size_t i = 1; i < args.size(); ++i){
		temp = add(temp, multiply(args[i], weights[i]));
	}
	temp = add(temp, bias);
	temp = activator(temp);
	return temp;
}


class Layer : public Module{
	std::vector<Neuron> neurons;
public:
	Layer(size_t input_size, size_t output_size, Value*(*activator)(Value*)):neurons{output_size, Neuron{input_size, 1, activator}} {}
	std::vector<Value*> getParams();
	std::vector<Value*> activate(std::vector<Value*>& args);
};


std::vector<Value*> Layer::getParams(){
	std::vector<Value*> res;
	for(auto n: neurons){
		auto v = n.getParams();
		res.insert(res.end(), v.begin(), v.end());
	}
	return res;
}


std::vector<Value*> Layer::activate(std::vector<Value*> &args){
	std::vector<Value*> res{neurons.size(), nullptr};
	for(size_t i = 0; i < neurons.size(); ++i)
		res[i] = neurons[i].activate(args);
	return res;
}

/*
class NeuralNet: public Module{
	std::vector<Layer> layers;
	std::vector<Value*> activate(std::vector<double> &args);
	Value*(*error)(std::vector<Value*>, std::vector<Value*>);
public:
	NeuralNet(size_t inputsize, std::vector<size_t> &shape, std::vector<Value*(*)(Value*)> &activators,Value*(*error)(std::vector<Value*>&, std::vector<Value*>&));
	std::vector<double> evaluate(std::vector<double>& args);
	void train(std::vector<double> &args, std::vector<double>& expected);
	std::vector<Value*> getParams();
};

NeuralNet::NeuralNet
(
	size_t inputsize,
	std::vector<size_t> &shape, 
	std::vector<Value*(*)(Value*)> &activators,
	Value*(*error)(std::vector<Value*>&, std::vector<Value*>&)
):
Module{},
layers{},
error{error}
{
	if(shape.size() != activators.size())
		throw std::runtime_error("Mismatch of layer count and activator count");
	// shape describes neuron count for each layer
	// which is the same as output size of layer
	layers.push_back(Layer(inputsize, shape[0], activators[0]));
	for(size_t i = 1; i < shape.size(); ++i){
		layers.push_back(Layer(shape[i - 1], shape[i], activators[i]));
	}
}


std::vector<Value*> NeuralNet::activate(std::vector<double> &args){
	std::vector<Value*> _args{args.size(), nullptr};
	for(size_t i = 0; i < _args.size(); ++i){
		_args[i] = new Value(args[i]);
	}
	for(auto layer: layers)
		_args = layer.activate(_args);
	return _args;
}


void NeuralNet::train(std::vector<double> &args, std::vector<double> &expected){
	std::vector<Value*> _expected{expected.size(), nullptr};
	for(size_t i = 0; i < expected.size(); ++i){
		_expected[i] = new Value(expected[i]);
	}
	
	std::vector<Value*> curr = activate(args);
	
	auto head = error(curr, _expected);
	head->backprop();
	handleMemory(head);
	descent();
	resetGrad();
}


std::vector<double> NeuralNet::evaluate(std::vector<double>& args){
	std::vector<Value*> temp = activate(args);
	std::vector<double> res{temp.size(), 0};
	for(size_t i = 0; i < res.size(); ++i){
		res[i] = temp[i]->val;
	}
	handleMemory(temp[0]);
	for(size_t i = 1; i < temp.size(); ++i){
		delete temp[i];
	}
	return res;
}


std::vector<Value*> NeuralNet::getParams(){
	std::vector<Value*> res;
	for(auto layer: layers){
		std::vector<Value*> lparams = layer.getParams();
		res.insert(res.end(), lparams.begin(), lparams.end());
	}
	return res;
}
*/

#endif
#ifndef __AUTODIFF_MICRO_H__
#define __AUTODIFF_MICRO_H__


#include <cmath>
#include <ostream>
#include <vector>
#include <iostream>
#include <algorithm>


template<typename T = double>
class Node{
	
public:
	// too lazy to call getters/setters everywhere
	T val;
	T grad;
	
	/*
	 * Backpropagation will be implemented in the following way: 
	 * Graph class will call those methods on all nodes in topological order (reverse topo for backprop)
	 * That is why these functions will not be recursive
	 */
	virtual void forward() = 0;
	virtual void backward() = 0;
	
	// constructors and =
	Node(T val = 0, T grad = 0): val(val), grad(grad) {}
	Node(Node &other) = default;
	Node& operator =(Node<T> &other) = default;
	Node(Node&&) = default;
	Node& operator =(Node<T>&&) = default;
	
	// dont forget virtual copy and destructor
	virtual Node<T>* clone() = 0;	
	virtual ~Node(){}
};


template<typename T = double>
class Value: public Node<T>{
public:
	Value() = default;
	Value(T val): Node<T>(val) {}
	Value(Value &other) = default;
	Value& operator =(Value<T> &other) = default;
	Value(Value&&) = default;
	Value& operator =(Value<T>&&) = default;
	void forward() override { this->grad = 0; }
	void backward() override {}
	virtual Value<T>* clone() { return new Value(*this); };
	
};

// graph will hold all nodes
// and is expected to handle all memory management
// copies share children, this is intended behavior
// TODO: 
template<typename T = double>
class UnaryExpression: public Node<T>{
protected:
	Node<T>* child;
public:
	UnaryExpression(Node<T>* child): child(child) {}
	void bindChild(Node<T> &node){
		child = &node;
	}
};


template<typename T = double>
class BinaryExpression: public Node<T>{
protected:
	Node<T>* left;
	Node<T>* right;
public:
	BinaryExpression(Node<T> *left, Node<T> *right): Node<T>(), left(left), right(right) {}
	void bindLeft(Node<T> &node){
		left = &node;
	}
	void bindRight(Node<T> &node){
		right = &node;
	}
};


template<typename T = double>
class Add: public BinaryExpression<T>{
public:
	Add(Node<T> *left, Node<T> *right):BinaryExpression<T>(left, right) {}
	
	void forward() override {
		this->grad = 0;
		this->val = this->left->val + this->right->val;
	}
	
	void backward() override {
		this->left->grad += this->grad;
		this->right->grad += this->grad;
	}
	
	
	virtual Add<T>* clone(){ return new Add<T>(*this); };
};


template<typename T = double>
class Sub: public BinaryExpression<T>{
public:
	Sub(Node<T> *left, Node<T> *right): BinaryExpression<T>(left, right) {}
	
	void forward() override {
		this->grad = 0;
		this->val = this->left->val - this->right->val;
	}
	
	void backward() override {
		this->left->grad += this->grad;
		this->right->grad -= this->grad;
	}
	
	virtual Sub<T>* clone(){ return new Sub<T>(*this); };
};


template<typename T = double>
class Mul: public BinaryExpression<T>{
public:
	Mul(Node<T> *left, Node<T> *right): BinaryExpression<T>(left, right) {}
	
	void forward() override {
		this->grad = 0;
		this->val = this->left->val * this->right->val;
	}
	
	void backward() override {
		this->left->grad += this->grad * this->right->val;
		this->right->grad += this->grad * this->left->val;
	}
	
	virtual Mul<T>* clone(){ return new Mul<T>(*this); };
};


template<typename T = double>
class Pow: public UnaryExpression<T>{
protected:
	int power;
public:
	Pow(Node<T> *child, int power): UnaryExpression<T>(child), power(power) {}
	
	void forward() override {
		this->grad = 0;
		this->val = pow(this->child->val, this->power);
	}
	
	void backward() override {
		this->child->grad = this->power * pow(this->child->val, this->power - 1);
	}
	
	virtual Pow<T>* clone(){ return new Pow<T>(*this); };
};


template<typename T = double>
class Div: public BinaryExpression<T>{
public:
	Div(Node<T> *left, Node<T> *right): BinaryExpression<T>(left, right) {}
	
	void forward() override {
		this->grad = 0;
		this->val = this->left->val / this->right->val;
	}
	
	void backward() override {
		this->left->grad += this->grad / this->right->val;
		this->right->grad += -1. / this->right->val / this->right->val * this->grad * this->left->val;
	}
	
	virtual Div<T>* clone(){ return new Div<T>(*this); };
};


template<typename T = double>
class Sigmoid: public UnaryExpression<T>{
protected:
	static double phi(double x){
		return 1 / (1 + std::exp(-x));
	}
public:
	Sigmoid(Node<T> *child): UnaryExpression<T>(child){}
	
	void forward() override {
		this->grad = 0;
		this->val = phi(this->child->val);
	}
	
	void backward() override {
		double temp = phi(this->child->val);
		this->child->grad += this->grad * temp * (1- temp);
	}
	
	virtual Sigmoid<T>* clone(){ return new Sigmoid<T>(*this); };
};

// this is redundant, leaving it for consistency
// specifically thinking about passing activator to neuron
// no harm in explicitly specifying linear (except some performance but oh well)
template<typename T = double>
class Linear: public UnaryExpression<T>{
public:
	Linear(Node<T> *child): UnaryExpression<T>(child){}
	
	void forward() override {
		this->grad = 0;
		this->val = this->child->val;
	}
	
	void backward() override {
		this->child->grad += this->grad;
	}
	
	virtual Linear<T>* clone(){ return new Linear<T>(*this); };
};


template<typename T = double>
class Relu: public UnaryExpression<T>{
public:
	Relu(Node<T> *child): UnaryExpression<T>(child) {}
	
	void forward() override {
		this->grad = 0;
		this->val = this->child->val > 0 ? this->child->val : 0;
	}
	
	void backward() override {
		this->child->grad += this->grad * (this->val > 0 ? 1 : 0);
	}
	
	virtual Relu<T>* clone(){ return new Relu<T>(*this); };
};


#endif
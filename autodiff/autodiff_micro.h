#ifndef __AUTODIFF_MICRO_H__
#define __AUTODIFF_MICRO_H__


#include <cmath>
#include <ostream>
#include <vector>
#include <iostream>
#include <algorithm>

// version for arbitrary input/output dimensions currently being developed
// it's almost ready but i can't be bothered to test/debug though
// armadillo's documentation is a pain to work with too
// that's why we go mini for now

// handling of dynamic memory issues is left to the reader as an exercise
// seriously though, memory deallocation must be handled by caller.
// TODO: copy constructor and assignment. copies currently share children 
// (as in the objects in the container, not the container itself) 
// due to how vector copy works.
struct Value{
public:
	double val;
	double grad;
	std::vector<Value*> children;
	void (*backward)(Value*);
	Value(){}
	Value(double val, std::vector<Value*> children = std::vector<Value*>(), void(*func)(Value*) = nullptr): val(val), grad(0), children(children), backward(func) {}
	void backprop();
	friend std::ostream& operator <<(std::ostream&, Value&);
};


std::ostream& operator <<(std::ostream& os, Value& v){
	os << "value - " << v.val << " gradient - " << v.grad << std::endl;
	for(auto child: v.children)
		os << *child;
	return os;
}


void add_back(Value* n){
	n->children[0]->grad += n->grad * 1;
	n->children[1]->grad += n->grad * 1;
}


Value* add(Value *a, Value *b){
	return new Value{a->val + b->val, std::vector<Value*>({a, b}), &add_back};
}


void subtract_back(Value* n){
	n->children[0]->grad += n->grad * 1;
	n->children[1]->grad += n->grad * (-1);
}


Value* subtract(Value *a, Value *b){
	return new Value{a->val - b->val, std::vector<Value*>({a, b}), &subtract_back};
}


void multiply_back(Value* n){
	n->children[0]->grad += n->grad * n->children[1]->val;
	n->children[1]->grad += n->grad * n->children[0]->val;
}


Value* multiply(Value *a, Value *b){
	
	return new Value{a->val * b->val, std::vector<Value*>({a, b}), &multiply_back};
}


void divide_back(Value* n){
	n->children[0]->grad += n->grad * (1 / n->children[1]->val);
	n->children[1]->grad += n->grad * (n->children[0]->val / (n->children[1]->val * n->children[1]->val));
}


Value* divide(Value *a, Value *b){
	if(b->val == 0)
		throw std::runtime_error("Division by zero");
	return new Value{a->val / b->val, std::vector<Value*>({a, b}), &divide_back};;
}


double phi(double x){
	return 1 / (1 + std::exp(-x));
}


void sigmoid_back(Value *n){
	double temp = phi(n->children[0]->val);
	n->children[0]->grad += n->grad * temp * (1- temp);
}


Value* sigmoid(Value *a){
	return new Value{phi(a->val), std::vector<Value*>({a}), &sigmoid_back};
}


void relu_back(Value* n){
	n->children[0]->grad += n->grad * (n->children[0]->val > 0 ? 1 : 0);
}


Value* relu(Value *a){
	return new Value{a->val > 0 ? a->val : 0, std::vector<Value*>({a}), &add_back};
}


// checks if val in v
template<typename T>
bool in(std::vector<T>& v, T& val){
	for(auto i : v){
		if(i == val)
			return true;
	}
	return false;
}

// topo sort, returns nodes in reverse topological order
// method assumes calculation graph is a DAG
// too lazy to implement safety checks, that's like two whole lines i need to write
void topo(Value *root, std::vector<Value*>& visited, std::vector<Value*>& res){
	for(auto child: root->children){
		if(!in(visited, child)){
			visited.push_back(child);
			topo(child, visited, res);
		}
	}
	res.push_back(root);
}


void Value::backprop(){
	// sort nodes
	std::vector<Value*> visited;
	std::vector<Value*> nodes;
	topo(this, visited, nodes);
	std::reverse(nodes.begin(), nodes.end());
	
	// update gradients
	this->grad = 1;
	for(auto v: nodes){
		if(v->backward != nullptr)
			v->backward(v);
	}
	
}


#endif
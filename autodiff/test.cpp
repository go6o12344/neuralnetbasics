#include "autodiff_micro.h"
#include <iostream>


int main(){
	/* 
		structure of example is the following:
				xy + sigmoid(z)
					/ \
				   /   \
				  /     \
				 /       \
				xy     sigmoid(z)
			   / \         \
			  /   \         \
			 x    y          z
		
		gradient is expected to be [y, x, sigmoid(z) * (1 - sigmoid(z))]
	*/
	//TODO: more testing
	double temp;
	std::cin >> temp;
	Value *x = new Value(temp);
	std::cin >> temp;
	Value *y = new Value(temp);
	std::cin >> temp;
	Value *z = new Value(temp);
	Value *xy = multiply(x, y);
	Value *sz = sigmoid(z);
	Value *xy_sz = add(xy, sz);
	// prints in the following order: xy_sz, xy, x, y, sz, z
	std::cout << *xy_sz;
	xy_sz->backprop();
	std::cout << "x: " << *x << "y: " << *y << "z: " << *z;
}
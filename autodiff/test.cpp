#include "autodiff_micro.h"
#include <iostream>


int main(){
	/* 
		structure is the following:
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
	double temp;
	std::cin >> temp;
	Value x(temp);
	std::cin >> temp;
	Value y(temp);
	std::cin >> temp;
	Value z(temp);
	Value xy = multiply(x, y);
	Value sz = sigmoid(z);
	Value xy_sz = add(xy, sz);
	std::cout << xy_sz;
	xy_sz.backprop();
	std::cout << "x: " << x << "y: " << y << "z: " << z;
}
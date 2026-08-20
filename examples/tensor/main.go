package main

import (
	"fmt"

	"github.com/coderianx/gonn"
)

func main() {
	// Create a 2x2 tensor
	X := gonn.NewTensor(
		[]float32{
			1, 2,
			3, 4,
		},
		2, 2,
	)

	// Print the shape of the tensor
	fmt.Println(X.Shape())
	// Print the data of the tensor
	fmt.Println(X.Data())
	// Print the size of the tensor
	fmt.Println(X.Size())
	// Print the number of dimensions of the tensor
	fmt.Println(X.Dim())
}

package main

import (
	"fmt"

	"github.com/coderianx/gonn"
)

func main() {
	// Datas
	X := gonn.NewTensor(
		[]float32{
			1,
			2,
			3,
			4,
			5,
		},
		5, 1,
	)

	y := gonn.NewTensor(
		[]float32{
			2,
			4,
			6,
			8,
			10,
		},
		5,
	)

	// Model
	model := gonn.NewLinearRegression()

	// Train
	model.Fit(
		X, y, 1000, 0.01,
	)

	// Predict
	predict := model.Predict(
		gonn.NewTensor([]float32{6}, 1, 1),
	)

	fmt.Println(predict)
	fmt.Println(predict.Data())

}

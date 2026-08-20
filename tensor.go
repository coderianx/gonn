package gonn

// Tensor represents a tensor (multi-dimensional array) of float32 values
type Tensor struct {
	data  []float32
	shape []int
}

// Shape returns the shape of the tensor
//
// Example:
//
//	t := gonn.NewTensor([]float32{1, 2, 3, 4}, 2, 2)
//	fmt.Println(t.Shape()) // [2 2]
func NewTensor(data []float32, shape ...int) *Tensor {
	return &Tensor{
		data:  data,
		shape: shape,
	}
}

// Shape returns the shape of the tensor
//
// Example:
//
//	t := gonn.NewTensor([]float32{1, 2, 3, 4}, 2, 2)
//	fmt.Println(t.Shape()) // [2 2]
func (t *Tensor) Shape() []int {
	shape := make([]int, len(t.shape))
	copy(shape, t.shape)

	return shape
}

// Data returns the data of the tensor
//
// Example:
//
//	t := gonn.NewTensor([]float32{1, 2, 3, 4}, 2, 2)
//	fmt.Println(t.Data()) // [1 2 3 4]
func (t *Tensor) Data() []float32 {
	data := make([]float32, len(t.data))
	copy(data, t.data)

	return data
}

// Size returns the size of the tensor
//
// Example:
//
//	t := gonn.NewTensor([]float32{1, 2, 3, 4}, 2, 2)
//	fmt.Println(t.Size()) // 4
func (t *Tensor) Size() int {
	return len(t.data)
}

// Dim returns the number of dimensions of the tensor
//
// Example:
//
//	t := gonn.NewTensor([]float32{1, 2, 3, 4}, 2, 2)
//	fmt.Println(t.Dim()) // 2
func (t *Tensor) Dim() int {
	return len(t.shape)
}

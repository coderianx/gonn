package gonn

type LinearRegression struct {
	weights *Tensor
	bias    float32
}

func NewLinearRegression() *LinearRegression {
	return &LinearRegression{}
}

func (m *LinearRegression) Fit(
	X *Tensor,
	y *Tensor,
	epochs int,
	lr float32,
) {
	samples := X.shape[0]
	features := X.shape[1]

	m.weights = NewTensor(
		make([]float32, features),
		features,
	)

	m.bias = 0

	for epoch := 0; epoch < epochs; epoch++ {
		weightGrad := make([]float32, features)
		var biasGrad float32

		for i := 0; i < samples; i++ {
			prediction := m.bias

			for j := 0; j < features; j++ {
				prediction += X.data[i*features+j] * m.weights.data[j]
			}

			err := prediction - y.data[i]

			for j := 0; j < features; j++ {
				weightGrad[j] += err * X.data[i*features+j]
			}

			biasGrad += err
		}

		for j := 0; j < features; j++ {
			m.weights.data[j] -= lr * weightGrad[j] / float32(samples)
		}

		m.bias -= lr * biasGrad / float32(samples)
	}
}

func (m *LinearRegression) Predict(X *Tensor) *Tensor {
	samples := X.shape[0]
	features := X.shape[1]

	result := make([]float32, samples)

	for i := 0; i < samples; i++ {
		prediction := m.bias

		for j := 0; j < features; j++ {
			prediction += X.data[i*features+j] * m.weights.data[j]
		}

		result[i] = prediction
	}

	return NewTensor(result, samples)
}

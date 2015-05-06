package svm

import (
	"errors"
)

type LinearKernelSVM struct {
	w      []float32
	alpha  float32
	lambda float32
}

func NewLinearKernelSVM(w []float32, alpha float32, lambda float32) *LinearKernelSVM {
	s := &LinearKernelSVM{}

	s.w = make([]float32, len(w))
	copy(s.w, w)

	s.alpha = alpha
	s.lambda = lambda

	return s
}

func (s *LinearKernelSVM) Predict(x []float32) (int, float32, error) {
	var y float32 = 0
	if len(s.w) != len(x) {
		return 0, 0.0, errors.New("len(s.w) != len(x)")
	}
	for i := 0; i < len(x); i++ {
		y += s.w[i] * x[i]
	}
	var t int = 0
	if y >= 0.0 {
		t = 1
	} else {
		t = -1
	}
	return t, y, nil
}

func (s *LinearKernelSVM) Train(x []float32, te int) {
	_, y, _ := s.Predict(x)
	if y*float32(te) < s.lambda {
		for i := 0; i < len(x); i++ {
			s.w[i] = s.w[i]*(1-s.alpha) + float32(te)*x[i]
		}
	}
}

func (s *LinearKernelSVM) Len() int {
	return len(s.w)
}

func (s *LinearKernelSVM) W(i int) float32 {
	return s.w[i]
}

func (s *LinearKernelSVM) Alpha() float32 {
	return s.alpha
}

func (s *LinearKernelSVM) Lambda() float32 {
	return s.lambda
}

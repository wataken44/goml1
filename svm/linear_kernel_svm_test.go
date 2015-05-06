package svm_test

import (
	"goml1/svm"
	"testing"
)

func TestPredict(t *testing.T) {
	var w []float32 = []float32{1.0, 2.0, -2.0}
	var x []float32 = []float32{1.0, 2.0, -3.0}

	s := svm.NewLinearKernelSVM(w, 0.1, 1.0)

	_, actual, _ := s.Predict(x)
	expected := float32(11.0)
	if actual != expected {
		t.Errorf("expected: %v, actual: %v", expected, actual)
	}
}

func TestTrain(t *testing.T) {
	var w []float32 = []float32{1.0, 1.0}
	var x []float32 = []float32{1.0, 1.0}

	s := svm.NewLinearKernelSVM(w, 0.25, 1.0)

	s.Train(x, 1)
	for i := 0; i < s.Len(); i++ {
		if s.W(i) != 1.0 {
			t.Errorf("expected: %v, actual: %v", 1.0, s.W(i))
		}
	}

	s.Train(x, -1)
	for i := 0; i < s.Len(); i++ {
		if s.W(i) != -0.25 {
			t.Errorf("expected: %v, actual: %v", -0.25, s.W(i))
		}
	}

}

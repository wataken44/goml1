
package perceptron_test

import (
	"testing"

	"goml1/perceptron"
)

func TestPredict(t *testing.T) {
	var w []float32 = []float32{1.0, 2.0, -2.0}
	var x []float32 = []float32{1.0, 2.0, -3.0}
	
	p := perceptron.NewPerceptron(w)

	_, actual, _ := p.Predict(x)
	expected := float32(11.0)
	if actual != expected {
		t.Errorf("expected: %v, actual: %v", expected, actual)
	}
}

func TestTrain(t *testing.T) {
	var w []float32 = []float32{1.0, 1.0}
	var x []float32 = []float32{1.0, 1.0}

	p := perceptron.NewPerceptron(w)

	p.Train(x, 1)
	for i := 0; i < p.Len(); i++ { 
		if p.W(i) != 1.0 {
			t.Errorf("expected: %v, actual: %v", 1.0, p.W(i))
		}
	}

	p.Train(x, -1)
	for i := 0; i < p.Len(); i++ { 
		if p.W(i) != 0.0 {
			t.Errorf("expected: %v, actual: %v", 0.0, p.W(i))
		}
	}

}


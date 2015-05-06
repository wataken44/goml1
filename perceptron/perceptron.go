package perceptron

import (
	"errors"
)

type Perceptron struct {
	w []float32
}

func NewPerceptron(w []float32) *Perceptron {
	var p *Perceptron = &Perceptron{}

	p.w = make([]float32, len(w))
	copy(p.w, w)
	return p
}

func (p *Perceptron) Predict(x []float32) (int, float32, error) {
	var y float32 = 0
	if len(p.w) != len(x) {
		return 0, 0.0, errors.New("len(p.w) != len(x)")
	}
	for i := 0; i < len(x); i++ {
		y += p.w[i] * x[i]
	}
	var t int = 0
	if y >= 0.0 {
		t = 1
	} else {
		t = -1
	}
	return t, y, nil
}

func (p *Perceptron) Train(x []float32, te int) {
	ta, _, _ := p.Predict(x)
	if ta != te {
		for i := 0; i < len(x); i++ {
			p.w[i] += float32(te) * x[i]
		}
	}
}

func (p *Perceptron) Len() int {
	return len(p.w)
}

func (p *Perceptron) W(i int) float32 {
	return p.w[i]
}

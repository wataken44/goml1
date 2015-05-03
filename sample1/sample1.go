
package main

import (
	"fmt"
	"math/rand"

	"goml1/perceptron"
)

func main() {
	fmt.Println("create perceptron which predict 10x + 15y > 1 for (x,y)")
	p := perceptron.NewPerceptron([]float32{0, 0, 0})

	fmt.Println("train with 10000 random data")
	for i := 0; i < 10000; i++ {
		// -1 <= x,y < 1
		x := float32((rand.Float32() - 0.5) * 2)
		y := float32((rand.Float32() - 0.5) * 2)
		t := -1
		if 10.0 * x + 15.0 * y > 1.0 {
			t = 1
		}
		p.Train([]float32{x, y, 1.0 /*bias*/}, t) 
	}

	fmt.Printf("current w = [")
	for i := 0; i < p.Len(); i++ {
		fmt.Printf("%f ", p.W(i))
	}
	fmt.Println("]")

	// predict
	fmt.Println("predict 10 random data")
	for i := 0; i < 10; i++ {
		// -1 <= x,y < 1
		x := float32((rand.Float32() - 0.5) * 2)
		y := float32((rand.Float32() - 0.5) * 2)
		te := -1
		if 10.0 * x + 15.0 * y > 1.0 {
			te = 1
		}
		ta, _, _ := p.Predict([]float32{x, y, 1.0})
		fmt.Printf("x = %f, y = %f, exp = %d, act = %d, ok = %v\n",
			x, y, te, ta, te == ta)
	}
	
}

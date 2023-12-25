package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
	_ "time"
)

type Weight struct {
	Value float64
}

type Bias struct {
	Value float64
}

type Node struct {
	Weights []Weight
	Bias    float64
}

type Layer struct {
	Nodes []Node
}

type Network struct {
	Layers []Layer
}

func (n Network) Randomize() {
	for i := range n.Layers {
		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				n.Layers[i].Nodes[j].Weights[k].Value = rand.NormFloat64()
			}
		}

		if i == 0 {
			continue
		}
		for j := range n.Layers[i].Nodes {
			n.Layers[i].Nodes[j].Bias = rand.NormFloat64()
		}
	}
}

func (n *Network) AppendLayer(width int) {
	var layer Layer

	for i := 0; i < width; i++ {
		layer.Nodes = append(layer.Nodes, Node{})
	}
	for i := range layer.Nodes {
		layer.Nodes[i].Bias = 0.0
		for j := 0; j < len(n.Layers[len(n.Layers)-1].Nodes); j++ {
			layer.Nodes[i].Weights = append(layer.Nodes[i].Weights, Weight{Value: 0.0})
		}
	}
	n.Layers = append(n.Layers, layer)
}

func (n Network) Forward(input []float64) (error, []float64) {
	if len(input) != len(n.Layers[0].Nodes) {
		return errors.New(fmt.Sprintf("Error parsing forward pass; Input size mismatch (input size: %d, input layer size: %d)", len(input), len(n.Layers[0].Nodes))), nil
	}
	prev := input
	curr := []float64{}
	for i := range n.Layers {
		if i == 0 {
			continue
		}

		curr = []float64{}
		for j := 0; j < len(n.Layers[i].Nodes); j++ {
			curr = append(curr, 0.0)
		}

		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				curr[j] += n.Layers[i].Nodes[j].Weights[k].Value * prev[k]

			}
			curr[j] += n.Layers[i].Nodes[j].Bias
			curr[j] = sigmoid(curr[j])
		}

		prev = curr
	}
	return nil, prev

}

func (n *Network) WeightGradient(weight *Weight, input []float64, expected []float64) (error, float64) {
	h := 0.001
	err, forwardOne := n.Forward(input)
	if err != nil {
		return errors.New("Error parsing weight gradient (forward pass failed)"), 0.0
	}
	err, costOne := cost(forwardOne, expected)
	if err != nil {
		return errors.New("Error parsing weight gradient (cost function failed)"), 0.0
	}
	weight.Value += h

	err, forwardTwo := n.Forward(input)
	if err != nil {
		return errors.New("Error parsing weight gradient (forward pass failed)"), 0.0
	}
	err, costTwo := cost(forwardTwo, expected)
	if err != nil {
		return errors.New("Error parsing weight gradient (cost function failed)"), 0.0
	}

	val := (costTwo - costOne) / h

	weight.Value -= h

	return nil, val
}

func (n *Network) BiasGradient(node *Node, input []float64, expected []float64) (error, float64) {
	h := 0.001
	err, forwardOne := n.Forward(input)
	if err != nil {
		return errors.New("Error parsing bias gradient (forward pass failed)"), 0.0
	}
	err, costOne := cost(forwardOne, expected)
	if err != nil {
		return errors.New("Error parsing bias gradient (cost function failed)"), 0.0
	}
	node.Bias += h

	err, forwardTwo := n.Forward(input)
	if err != nil {
		return errors.New("Error parsing bias gradient (forward pass failed)"), 0.0
	}
	err, costTwo := cost(forwardTwo, expected)
	if err != nil {
		return errors.New("Error parsing bias gradient (cost function failed)"), 0.0
	}

	val := (costTwo - costOne) / h

	node.Bias -= h

	return nil, val
}

func (n *Network) Train(input []float64, expected []float64) error {
	for i := range n.Layers {
		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				err, grad := n.WeightGradient(&n.Layers[i].Nodes[j].Weights[k], input, expected)
				if err != nil {
					return errors.New("Error training network (gradient failed)")
				}
				n.Layers[i].Nodes[j].Weights[k].Value -= grad
			}

			err, grad := n.BiasGradient(&n.Layers[i].Nodes[j], input, expected)
			if err != nil {
				return errors.New("Error training network (gradient failed)")
			}
			n.Layers[i].Nodes[j].Bias -= grad

		}
	}

	return nil
}

func (n *Network) TrainingPass(inputData [][]float64, expectedData [][]float64, passes int) error {
	if len(inputData) != len(expectedData) {
		return errors.New("Error in training pass; input size mismatch")
	}
	if len(inputData[0]) != len(n.Layers[0].Nodes) {
		fmt.Println("yoooo")
		return errors.New("Error in training pass; input width mismatch")
	}
	if len(expectedData[0]) != len(n.Layers[len(n.Layers)-1].Nodes) {
		return errors.New("Error in training pass; output width mismatch")
	}

	for i := 0; i < passes; i++ {
		randomInt := rand.Intn(len(inputData))
		n.Train(inputData[randomInt], expectedData[randomInt])
	}

	return nil
}

func sigmoid(input float64) float64 {
	return 1 / (1 + math.Exp(-input))
}

func cost(realized []float64, expected []float64) (error, float64) {
	if len(realized) != len(expected) {
		return errors.New("Error parsing cost; Input size mismatch"), 0.0
	}

	sum := 0.0
	for i, v := range realized {
		sum += (v - expected[i]) * (v - expected[i])
	}

	return nil, sum

}

func initNetwork(width int) Network {
	var network Network
	network.Layers = append(network.Layers, Layer{})

	for i := 0; i < width; i++ {
		network.Layers[0].Nodes = append(network.Layers[0].Nodes, Node{})
	}

	return network
}

func parseCsv(path string) (error, [][]float64) {
	file, err := os.Open(path)
	if err != nil {
		return errors.New("Error opening csv file"), nil
	}

	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		return errors.New("Error reading csv file"), nil
	}

	records = records[1:]

	data := make([][]float64, len(records))

	for i, record := range records {
		row := make([]float64, len(record))
		for j, value := range record {
			num, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return errors.New("Error converting string to float"), nil
			}
			row[j] = num
		}
		data[i] = row
	}

	return nil, data
}

func parseData(data [][]float64) ([][]float64, [][]float64) {
	labels := []float64{}
	for i := range data {
		labels = append(labels, data[i][0])
		data[i] = data[i][1:]
		for j := range data[i] {
			// data[i][j] = data[i][j] / 256.0
			j = j
			continue
		}
	}

	input := make([][]float64, len(labels))
	for i := range labels {
		input[i] = []float64{0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0}

		input[i][int(labels[i]*10)-1] = 1.0
	}

	return data, input
}

func main() {
	// rand.Seed(1)
	timeOne := time.Now().Unix()
	var network Network = initNetwork(10)
	network.AppendLayer(20)
	network.AppendLayer(10)
	network.Randomize()
	fmt.Println(time.Now().Unix() - timeOne)
	//network.Randomize()

	/*
		err, data := parseCsv("test.csv")
		if err != nil {
			fmt.Println(err)
		}

		err, test := parseCsv("test.csv")
		if err != nil {
			fmt.Println(err)
		}

		trainInput, trainOutput := parseData(data)
		fmt.Println(trainInput, trainOutput)
		testInput, testOutput := parseData(test)
		fmt.Println(testInput, testOutput)
	*/

	trainInput := [][]float64{
		[]float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
	}

	trainOutput := [][]float64{
		[]float64{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
		[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
		[]float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	}

	err := network.TrainingPass(trainInput, trainOutput, 1000)
	if err != nil {
		fmt.Println(err)
	}

	err, forward := network.Forward(trainInput[9])
	if err != nil {
		fmt.Println(err)
	}
	fmt.Print("forward: ")
	fmt.Println(forward)

	err, cost := cost(forward, trainOutput[9])
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(fmt.Sprintf("cost: %f", cost))
	// fmt.Println(network)
}

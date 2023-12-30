package main

import (
	_"strconv"
	_"bufio"
	_"os"
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	_ "math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
	_ "time"
	"encoding/json"
	"io/ioutil"
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

type NodePosition struct {
	Node  Node
	Index int	
}

type LayerPosition struct {
	Layer Layer
	Index int
}

type Data struct {
	InputData [][]float64
	OutputData [][]float64
}

func (n Node) Copy() Node {
	node := Node{}

	for i := range n.Weights {
		node.Weights = append(node.Weights, Weight{Value: n.Weights[i].Value})
	}
	node.Bias = n.Bias

	return node
}

func (l Layer) Copy() Layer {
	layer := Layer {}
	for i := range l.Nodes {
		layer.Nodes = append(layer.Nodes, Node{})
		for j := range l.Nodes[i].Weights {
			layer.Nodes[i].Weights = append(layer.Nodes[i].Weights, Weight{Value: l.Nodes[i].Weights[j].Value})
		}
		layer.Nodes[i].Bias = l.Nodes[i].Bias
	}
	return layer
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

func (n *Network) WeightGradient(weight *Weight, input []float64, expected []float64, learningRate float64) (error, float64) {
	h := 0.000001
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

	weight.Value -= val * learningRate

	return nil, val
}

func (n *Network) BiasGradient(node *Node, input []float64, expected []float64, learningRate float64) (error, float64) {
	h := 0.000001
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

	node.Bias -= val * learningRate

	return nil, val
}

/*
func (n *Network) ParallelTrainLayers(input []float64, expected []float64, layer *Layer, layerChannel chan<- LayerPosition, wg *sync.WaitGroup, index int) error {
	defer wg.Done()
	for j := range layer.Nodes {
		for k := range layer.Nodes[j].Weights {
			err, grad := n.WeightGradient(&layer.Nodes[j].Weights[k], input, expected)
			if err != nil {
				return errors.New("Error training network (gradient failed)")
			}
			layer.Nodes[j].Weights[k].Value -= grad
		}

		err, grad := n.BiasGradient(&layer.Nodes[j], input, expected)
		if err != nil {
			return errors.New("Error training network (gradient failed)")
		}
		layer.Nodes[j].Bias -= grad
	}

	layerChannel <- LayerPosition{Layer: *layer, Index: index}

	return nil
}
*/

func (n *Network) ParallelTrainNodes(input []float64, expected []float64, node *Node, nodeChannel chan<- NodePosition, wg *sync.WaitGroup, index int, learningRate float64) error {
	defer wg.Done()

	for i := range node.Weights {
		err, grad := n.WeightGradient(&node.Weights[i], input, expected, learningRate)
		if err != nil {
			return errors.New("Error training network (gradient failed)")
		}
		node.Weights[i].Value -= grad
	}

	err, grad := n.BiasGradient(node, input, expected, learningRate)
	if err != nil {
		return errors.New("Error training network (gradient failed)")
	}
	node.Bias -= grad

	nodeChannel <- NodePosition{Node: *node, Index: index}

	return nil
}

func (n *Network) ParallelTrainLayers(input []float64, expected []float64, layer *Layer, layerChannel chan<- LayerPosition, wg1 *sync.WaitGroup, index int, learningRate float64) error {
	defer wg1.Done()
	var wg2 sync.WaitGroup
	nodeChannel := make(chan NodePosition, len(layer.Nodes))
	for i := range layer.Nodes {
		network := n.Copy()
		wg2.Add(1)
		go network.ParallelTrainNodes(input, expected, &network.Layers[index].Nodes[i], nodeChannel, &wg2, i, learningRate)
	}

	wg2.Wait()
	close(nodeChannel)

	for i := range nodeChannel {
		layer.Nodes[i.Index] = i.Node.Copy()
	}

	layerChannel <- LayerPosition{Layer: *layer, Index: index}

	return nil
}


func (n *Network) ParallelTrain(input []float64, expected []float64, networkChannel chan<- Network, wg1 *sync.WaitGroup, leaningRate float64) error {
	defer wg1.Done()
	var wg2 sync.WaitGroup
	layerChannel := make(chan LayerPosition, len(n.Layers))

	for i := range n.Layers {
		network := n.Copy()
		wg2.Add(1)
		go network.ParallelTrainLayers(input, expected, &network.Layers[i], layerChannel, &wg2, i, leaningRate)
	}

	wg2.Wait()
	close(layerChannel)
	for i := range layerChannel {
		n.Layers[i.Index] = i.Layer.Copy()
	}
	/*
	// A possible asynchronous solution to assembling the new network layers
	for i := range layerChannel {
		// fmt.Println(len(layerChannel))
		wg.Add(1)
		go func() {
			defer wg.Done()
			i := i
			n.Layers[i.Index] = i.Layer.Copy()
		}()
		// fmt.Println(i.Index)
	}
	wg.Wait()
	*/

	networkChannel <- n.Copy()

	return nil
}

func (n *Network) Train(inputs [][]float64, expecteds [][]float64, learningRate float64) error {
	if len(inputs) != len(expecteds) {
		return errors.New("Error in train; input width mismatch")
	}

	var wg sync.WaitGroup
	networkChannel := make(chan Network, len(inputs))

	timeOne := time.Now().UnixMilli()
	for i := range inputs {
		network := n.Copy()
		wg.Add(1)
		go network.ParallelTrain(inputs[i], expecteds[i], networkChannel, &wg, learningRate)
	}
	

	wg.Wait()
	close(networkChannel)
	fmt.Println("duration:", time.Now().UnixMilli() - timeOne, "ms")

	*n = n.Zeros()
	networkChannelLength := 0

	for i := range networkChannel {
		networkChannelLength++
		n.Add(i)
	}
	n.ScalarMultiply(1.0 / float64(networkChannelLength))
	
	return nil
}

/*
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
}*/

func (n *Network) TrainingPass(inputData [][]float64, expectedData [][]float64, passes int, batchSize int, learningRate float64) error {
	if len(inputData) != len(expectedData) {
		return errors.New("Error in training pass; input size mismatch")
	}
	if len(inputData[0]) != len(n.Layers[0].Nodes) {
		return errors.New("Error in training pass; input width mismatch")
	}
	if len(expectedData[0]) != len(n.Layers[len(n.Layers)-1].Nodes) {
		return errors.New("Error in training pass; output width mismatch")
	}

	for i := 0; i < passes; i++ {
		fmt.Println("Pass: ", i)
		randomInt := rand.Intn(len(inputData))
		for {
			if (len(inputData) - randomInt) >= batchSize {
				break
			}
			randomInt = rand.Intn(len(inputData))
		}

		//fmt.Println(n)
		n.Train(inputData[randomInt:randomInt + batchSize], expectedData[randomInt:randomInt + batchSize], learningRate)
		_, forward := n.Forward(inputData[randomInt])
		_, delta := cost(forward, expectedData[randomInt])
		fmt.Println("cost: ", delta)
	}

	return nil
}

func (n *Network) TestingPass(inputData [][]float64, expectedData [][]float64, passes int) (error, float64) {
	if len(inputData) != len(expectedData) {
		return errors.New("Error in testing pass; input size mismatch"), 0.0
	}
	if len(inputData[0]) != len(n.Layers[0].Nodes) {
		return errors.New("Error in testing pass; input width mismatch"), 0.0
	}
	if len(expectedData[0]) != len(n.Layers[len(n.Layers)-1].Nodes) {
		return errors.New("Error in testing pass; output width mismatch"), 0.0
	}

	var sum float64

	for i := 0; i < passes; i++ {
		randomInt := rand.Intn(len(inputData))

		err, delta := n.Test(inputData[randomInt], expectedData[randomInt])

		if err != nil {
			return errors.New("Error in testing pass (test failed)"), 0.0
		}

		sum += delta
	}

	return nil, sum / float64(passes)
}

func (n *Network) Test(inputs []float64, expecteds []float64) (error, float64) {
	err, forward := n.Forward(inputs)
	err, delta := cost(forward, expecteds)

	if err != nil {
		return errors.New("Error testing (cost function failed)"), 0.0
	}

	return nil, delta
}


func (n Network) Copy() Network {
	network := initNetwork(len(n.Layers[0].Nodes))

	for i := range n.Layers {
		if i == 0 {
			continue
		}

		network.AppendLayer(len(n.Layers[i].Nodes))
		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				network.Layers[i].Nodes[j].Weights[k] = n.Layers[i].Nodes[j].Weights[k]
			}

			network.Layers[i].Nodes[j].Bias = n.Layers[i].Nodes[j].Bias
		}
	}

	return network
}

func (n *Network) Average(network Network) {
	for i := range n.Layers {
		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				n.Layers[i].Nodes[j].Weights[k].Value = (n.Layers[i].Nodes[j].Weights[k].Value + network.Layers[i].Nodes[j].Weights[k].Value) / 2
			}
			n.Layers[i].Nodes[j].Bias = (n.Layers[i].Nodes[j].Bias + network.Layers[i].Nodes[j].Bias) / 2
		}
	}
}

func (n *Network) Add(network Network) {
	for i := range n.Layers {
		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				n.Layers[i].Nodes[j].Weights[k].Value += network.Layers[i].Nodes[j].Weights[k].Value
			}
			n.Layers[i].Nodes[j].Bias += network.Layers[i].Nodes[j].Bias
		}
	}
}

func (n *Network) Multiply(network Network) {
	for i := range n.Layers {
		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				n.Layers[i].Nodes[j].Weights[k].Value *= network.Layers[i].Nodes[j].Weights[k].Value
			}
			n.Layers[i].Nodes[j].Bias *= network.Layers[i].Nodes[j].Bias
		}
	}
}

func (n *Network) ScalarMultiply(scalar float64) {
	for i := range n.Layers {
		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				n.Layers[i].Nodes[j].Weights[k].Value = n.Layers[i].Nodes[j].Weights[k].Value * scalar
			}
			n.Layers[i].Nodes[j].Bias = n.Layers[i].Nodes[j].Bias * scalar
		}
	}
}

func (n Network) Zeros() Network{
	network := n.Copy()

	for i := range n.Layers {
		for j := range n.Layers[i].Nodes {
			for k := range n.Layers[i].Nodes[j].Weights {
				network.Layers[i].Nodes[j].Weights[k].Value = 0
			}
			network.Layers[i].Nodes[j].Bias = 0
		}
	}

	return network
}

func (n Network) Pack(path string) error {
	jsonData, err := json.MarshalIndent(n, "", "	")
	if err != nil {
		return errors.New("Error packing network (json marshalling failed)")
	}

	err = ioutil.WriteFile(path, jsonData, 0644)
	if err != nil {
		return errors.New("Error packing network (file write failed)")
	}

	return nil
}

func unpack(path string) (error, Network) {
	fileData, err := ioutil.ReadFile(path)
	if err != nil {
		return errors.New("Error unpacking network (file read failed)"), Network{}
	}

	var network Network

	err = json.Unmarshal(fileData, &network)
	if err != nil {
		return errors.New("Error unpacking network (json unmarshalling failed)"), Network{}
	}

	return nil, network
}

func sigmoid(input float64) float64 {
	// return (math.Exp(input) - math.Exp(-input)) / (math.Exp(-input) - math.Exp(input))
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

func parseData(data [][]float64, splitColumn int) (error, [][]float64, [][]float64) {
	trainInput := [][]float64{}
	trainExpected := [][]float64{}
	if splitColumn >= len(data[0]) {
		return errors.New("Failed parsing data (split column index is too large)"), nil, nil
	}

	for i := range data {
		trainInput = append(trainInput, []float64{})
		trainExpected = append(trainExpected, []float64{})
		for j := range data[i] {
			if j >= splitColumn {
				trainExpected[i] = append(trainExpected[i], data[i][j])
				continue
			}
			trainInput[i] = append(trainInput[i], data[i][j])
		}
	}

	return nil, trainInput, trainExpected
}

func (n *Network) CommandInitialize() {
	var size string
	var network Network

	fmt.Println("Enter the amount of layers")
	fmt.Scanln(&size)

	networkSize, _ := strconv.Atoi(size)

	fmt.Println("Enter the size of each layer in order")

	for i := 0; i < networkSize; i++ {
		layersString := ""
		fmt.Scanln(&layersString)
		layer, _ := strconv.Atoi(layersString)

		if i == 0 {
			network = initNetwork(layer)
			continue
		}

		network.AppendLayer(layer)
	}

	network.Randomize()

	*n = network.Copy()
}

func (n *Network) CommandTrain(trainData Data) {
	var batchSize string
	var passesSize string
	var learningRateSize string

	fmt.Println("Enter the batch size")
	fmt.Scanln(&batchSize)
	fmt.Println("Enter the amount of training passes")
	fmt.Scanln(&passesSize)
	fmt.Println("Enter the learning rate")
	fmt.Scanln(&learningRateSize)

	batches, _ := strconv.Atoi(batchSize)
	passes, _ := strconv.Atoi(passesSize)
	learningRate, _ := strconv.ParseFloat(learningRateSize, 64)

	err := n.TrainingPass(trainData.InputData, trainData.OutputData, passes, batches, learningRate)
	if err != nil {
		fmt.Println(err)
	}
}

func (a *Data) CommandLoadCsv() {
	var path string
	var splitColumn string

	fmt.Println("Enter the file path")
	fmt.Scanln(&path)

	err, data := parseCsv(path)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("Enter the column to split the csv into expected and input data (for example entering 0 means that all the data from column 0 onwards is input data")
	fmt.Scanln(&splitColumn)
	column, _ := strconv.Atoi(splitColumn)
	err, output, input := parseData(data, column)
	if err != nil {
		fmt.Println(err)
	}

	a.InputData = input
	a.OutputData = output
}

func (n *Network) CommandTest(testData Data) {
	var passesSize string

	fmt.Println("Enter the amount of testing passes")
	fmt.Scanln(&passesSize)

	passes, _ := strconv.Atoi(passesSize)

	err, result := n.TestingPass(testData.InputData, testData.OutputData, passes)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("Network testing average cost:", result)
}

func (n *Network) CommandPack() {
	var path string

	fmt.Println("Enter the path to pack the network to")
	fmt.Scanln(&path)

	err := n.Pack(path)
	if err != nil {
		fmt.Println(err)
	}
}

func (n *Network) CommandUnpack() {
	var path string
	fmt.Println("Enter the path to unpack the network from")
	fmt.Scanln(&path)

	err, network := unpack(path)
	if err != nil {
		fmt.Println(err)
	}

	*n = network.Copy()
}

func main() {
	fmt.Println("Welcome to the simple go neural network trainer")
	fmt.Println("-----------------------------------------------")
	fmt.Println("Enter 1 to initialize a new network")
	fmt.Println("Enter 2 to load CSV-file for training")
	fmt.Println("Enter 3 to load CSV-file for testing")
	fmt.Println("Enter 4 to train using loaded CSV-file")
	fmt.Println("Enter 5 to test using loaded CSV-file")
	fmt.Println("Enter 6 to pack loaded network")
	fmt.Println("Enter 7 to unpack and load a network")

	var mode string
	var n Network
	var trainData Data
	var testData Data

	for {
		fmt.Scanln(&mode)

		if mode == "1" {
			n.CommandInitialize()
			fmt.Println(n)
		}
		if mode == "2" {
			trainData.CommandLoadCsv()
		}
		if mode == "3" {
			testData.CommandLoadCsv()
		}
		if mode == "4" {
			n.CommandTrain(trainData)
		}
		if mode == "5" {
			n.CommandTest(testData)
		}
		if mode == "6" {
			n.CommandPack()
		}
		if mode == "7" {
			n.CommandUnpack()
		}

		fmt.Println("Enter 1 to initialize a new network")
		fmt.Println("Enter 2 to load CSV-file for training")
		fmt.Println("Enter 3 to load CSV-file for testing")
		fmt.Println("Enter 4 to train using loaded CSV-file")
		fmt.Println("Enter 5 to test using loaded CSV-file")
		fmt.Println("Enter 6 to pack loaded network")
		fmt.Println("Enter 7 to unpack and load a network")
	}
}

/*
	// rand.Seed(1)
	var network Network = initNetwork(1)
	network.AppendLayer(200)
	network.AppendLayer(1)
	network.Randomize()
	
	err, data := parseCsv("train.csv")
	if err != nil {
		fmt.Println(err)
	}

	err, trainInput, trainOutput := parseData(data, 1)
	if err != nil {
		fmt.Println(err)
	}

	timeOne := time.Now().Unix()
	err = network.TrainingPass(trainInput, trainOutput, 1000, 1)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("-------------------------------")
	fmt.Println("training time:", time.Now().Unix() - timeOne, "seconds")

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
*/
	/*trainInput := [][]float64{
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
	}

	trainOutput := [][]float64{
		{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
		{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
	}
	*/

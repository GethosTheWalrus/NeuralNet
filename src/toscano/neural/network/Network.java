package toscano.neural.network;

public class Network {
	
	private int numNeurons;
	
	// global neurons
	private Neuron[] neurons; // hidden
	private Neuron o; // output
	
	public Network(int numNeurons, int inputSize) {
		
		this.numNeurons = numNeurons;
		this.initialize(inputSize);
		
	}
	
	private void initialize(int inputSize) {
				
		// Creating the neurons
		this.neurons = new Neuron[this.numNeurons];
		for(int i = 0; i < this.numNeurons; i++) {
			
			Neuron n = new Neuron(inputSize);
			this.neurons[i] = n;
			
		}
		
		o = new Neuron(this.numNeurons);
		
		// Assign random weights
		for(int i = 0; i < this.numNeurons; i++) {
			
			this.neurons[i].randomizeWeights();
			
		}
		
		o.randomizeWeights();
		
	}
	
	public double predict(double[] inputs) {
		
		for( int n = 0; n < this.numNeurons; n++ ) {
							
			// each input value in this set of inputs (e.g. loop through inputs[0] to get inputs[0][0], [0][1] .... [0][p]
			for( int p = 0; p < inputs.length; p++ ) {
				
				this.neurons[n].inputs[p] = inputs[p];
				
			}
			
		}
		
		// set the output neuron's inputs to the activated values of the hidden layer neurons
		for(int n = 0; n < this.numNeurons; n++) {
			
			o.inputs[n] = this.neurons[n].output();
			
		}
		
		return o.output();
		
	}
	
	private void learn(double expected) {
		
		// adjust the weight of the output neuron based on its error
		o.error = Sigmoid.derivative(o.output()) * (expected - o.output());
		o.adjustWeights();
		
		// adjust the weights of the hidden neurons based on their errors
		for(int n = 0; n < this.numNeurons; n++) {
			
			this.neurons[n].error = Sigmoid.derivative(this.neurons[n].output()) * o.error * o.weights[n];
			
		}
		
		for(int n = 0; n < this.numNeurons; n++) {
			
			this.neurons[n].adjustWeights();
			
		}
		
	}
	
	public void train(int numSessions, double[][] inputs, double[] expected) {
		
		// i training sessions
		for( int i = 0; i < numSessions; i++ ) {
			
			// train each possible input once per session
			for( int j = 0; j < inputs.length; j++ ) {
				
				// 1) forward propagation to calculate output
				this.predict(inputs[j]);
				
				// 2) back propatation to adjust weights
				this.learn(expected[j]);
				
			}
			
		}
		
	}
	
}
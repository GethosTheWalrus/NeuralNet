package toscano.neural.network;

public class Network {
	
	private int numNeurons;
	private Neuron[] neurons;
	
	public Network(int numNeurons) {
		
		this.numNeurons = numNeurons;
		
	}
	
	public void train() {
		
		// Possible inputs
		double[][] inputs = {
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};
		
		// Expected outputs
		double[] expected = {0, 1, 1, 0};
		
		// Creating the neurons
		this.neurons = new Neuron[this.numNeurons];
		for(int i = 0; i < this.numNeurons; i++) {
			
			Neuron n = new Neuron(inputs.length);
			this.neurons[i] = n;
			
		}
		
		Neuron o = new Neuron(this.numNeurons);
		
		// Assign random weights
		for(int i = 0; i < this.numNeurons; i++) {
			
			this.neurons[i].randomizeWeights();
			
		}
		
		o.randomizeWeights();
		
		// i training sessions
		for( int i = 0; i < 10000; i++ ) {
			
			// train each possible input once per session
			for( int j = 0; j < inputs.length; j++ ) {
				
				// 1) forward propagation to calculate output
				for( int n = 0; n < this.numNeurons; n++ ) {
									
					// each input value in this set of inputs (e.g. loop through inputs[0] to get inputs[0][0], [0][1] .... [0][p]
					for( int p = 0; p < inputs[j].length; p++ ) {
						
						this.neurons[n].inputs[p] = inputs[j][p];
						
					}
					
				}
				
				// set the output neuron's inputs to the activated values of the hidden layer neurons
				for(int n = 0; n < this.numNeurons; n++) {
					
					o.inputs[n] = this.neurons[n].output();
					
				}
				
				System.out.println(String.format("%f xor %f = %f", inputs[j][0], inputs[j][1], o.output()));
				
				// 2) back propatation to adjust weights
				
				// adjust the weight of the output neuron based on its error
				o.error = Sigmoid.derivative(o.output()) * (expected[j] - o.output());
				o.adjustWeights();
				
				// adjust the weights of the hidden neurons based on their errors
				for(int n = 0; n < this.numNeurons; n++) {
					
					this.neurons[n].error = Sigmoid.derivative(this.neurons[n].output()) * o.error * o.weights[n];
					
				}
				
				for(int n = 0; n < this.numNeurons; n++) {
					
					this.neurons[n].adjustWeights();
					
				}
				
			}
			
		}
		
	}
	
}
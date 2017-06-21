package toscano.neural.network;

public class Main {
	
	public static void main(String[] args) {
		
		// Possible inputs
		double[][] inputs = {
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};
		
		// Expected outputs
		double[] expected = {0, 1, 1, 0};
		
		// Network parameters
		int numNeurons = 3;
		
		System.out.println(String.format("Creating new neural network with %d neurons expecting input of length %d", numNeurons, inputs[0].length));
		Network n = new Network(numNeurons, inputs[0].length);
		
		System.out.println("Training network...");
		n.train(10000000, inputs, expected);
		
		System.out.println(String.format("Network fully trained with error %f\n", n.getError()));
		
		for( int i = 0; i < inputs.length; i++ ) {
			
			System.out.println(String.format("%d xor %d prediction = %f. Actual value = %d", (int)inputs[i][0], (int)inputs[i][1], n.predict(inputs[i]), (int)expected[i]));
			
		}
		
	}

}

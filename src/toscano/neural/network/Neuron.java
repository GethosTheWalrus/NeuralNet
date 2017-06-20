package toscano.neural.network;

import java.util.Random;

public class Neuron {
	
	public double[] inputs;
	public double[] weights;
	public double error;
	
	private double biasWeight;
	private int numInputs;
	
	private Random r = new Random();
	
	public Neuron(int numInputs) {
		
		this.numInputs = numInputs;
		this.inputs = new double[numInputs];
		this.weights = new double[numInputs];
		
	}
	
	public double output() {
		
		return Sigmoid.output(weights[0] * inputs[0] + weights[1] * inputs[1] + biasWeight);
		
	}
	
	public void randomizeWeights() {
		
		for(int i = 0; i < this.numInputs; i++) {
			
			weights[i] = this.r.nextDouble();
			
		}
		
		biasWeight = r.nextDouble();
		
	}
	
	public void adjustWeights() {
		
		for(int i = 0; i < this.numInputs; i++) {
			
			weights[i] += this.error * inputs[i];
			
		}
		
		biasWeight += error;
		
	}

}

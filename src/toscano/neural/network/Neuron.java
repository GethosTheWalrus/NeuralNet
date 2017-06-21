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
		
		double expression = 0;
		for( int i = 0; i < this.numInputs; i++ ) {
			
			expression += this.weights[i] * this.inputs[i];
			
		}
		expression += this.biasWeight;
		
		return Sigmoid.output(expression);
		
	}
	
	public void randomizeWeights() {
		
		for(int i = 0; i < this.numInputs; i++) {
			
			this.weights[i] = this.r.nextDouble();
			
		}
		
		this.biasWeight = r.nextDouble();
		
	}
	
	public void adjustWeights() {
		
		for(int i = 0; i < this.numInputs; i++) {
			
			this.weights[i] += this.error * this.inputs[i];
			
		}
		
		this.biasWeight += error;
		
	}

}

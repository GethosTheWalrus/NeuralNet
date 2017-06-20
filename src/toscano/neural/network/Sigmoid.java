package toscano.neural.network;

public class Sigmoid {
	
	public static double output(double x) {
		
		return 1.0 / (1.0 + Math.exp(-x));
		
	}
	
	public static double derivative(double x) {
		
		return x * (1 - x);
		
	}

}

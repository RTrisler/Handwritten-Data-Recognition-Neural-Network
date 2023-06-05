package hdrnueralnetwork;


/* Network Config class that contains variables for configurable parts of the network*/
public class NetworkConfig 
{
	public int epochs;
	public int batchSize;
	public double learningRate;
	
	public NetworkConfig()
	{
		epochs = 30;
		batchSize = 10;
		learningRate = 3;
	}
}

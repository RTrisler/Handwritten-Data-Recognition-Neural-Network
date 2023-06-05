/* Reid Trisler
 * CWID: 10330107
 * 10/27/2022
 * CSC 475 Assignment #2 MNIST Handwritten Digit Neural Network in Java 
 * 
 * Fully Connected Feed Forward Neural Network trained using back propagation and stochastic gradient descent
 * The user can select to train a new network or load in a pretrained network
 * The Network Loads in data from the MNIST data set and begins training over the 60k classified images provided.
 * The network updated the weights and biases after each mini batch using the process of back propagation.
 * After the network is trained the user can display the accuracy of the network against the training and testing sets as well as save the weights and biases to a file.
 * 
 * 
 * 
 * 
 * 
 * 
 * */




package hdrnueralnetwork;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;



public class Network
{
	// initialize variables and arrays
	private static int training_data_size = 60000;
	private static int testing_data_size = 10000;
	private int numLayers;
	private int[]layer_sizes;
	private static double[][] label_set;
	private static double[][] label_set_testing;
	private static double[][] inputArrays;
	private static double[][] inputArrays_testing;
	//initialize matrices
	private static Matrix weights_input_to_hidden, weights_hidden_to_output, bias_hidden, bias_output;
	
	// initialize 2D array of Matrices used for storing the gradients per mini batch so they can be used in the update methods 
	private static Matrix[] biasgradients_per_batch_output;
	private static Matrix[] biasgradients_per_batch_hidden;
	private static Matrix[] weightgradients_per_batch_ho;
	private static Matrix[] weightgradients_per_batch_ih;
	
	private NetworkConfig config;
	
	// keeps track of accuracy after each epoch
	public static List<Accuracy> AccuracyList = new ArrayList<>();
	
	// constructor takes in int array that holds the size for each layer and the configuration
	// instantiating variables
	public Network(int[] layer_sizes, NetworkConfig config)
	{
		this.config = config;
		this.numLayers = layer_sizes.length;
		this.layer_sizes = layer_sizes; 
		
		// 60k x 10
		// each row holds the classification of the image in one hot vector format
		label_set = new double[training_data_size][layer_sizes[2]];
		// 10k by 10
		label_set_testing = new double[testing_data_size][layer_sizes[2]];
		// 60k by 784
		// each row holds all pixels for one image
		inputArrays = new double[training_data_size][layer_sizes[0]];
		// 10k by 784
		inputArrays_testing = new double[testing_data_size][layer_sizes[0]];
		
		
		// 30 x 784
		weights_input_to_hidden = new Matrix(layer_sizes[1], layer_sizes[0]);
		// 10 x 30
		weights_hidden_to_output = new Matrix(layer_sizes[2], layer_sizes[1]);
		
		// 30 x 1
		bias_hidden = new Matrix(layer_sizes[1], 1);
		// 10 x 1
		bias_output = new Matrix(layer_sizes[2], 1);
		
		// 1D array of Matrices  holding at each index another matrix containing the gradients for specific image in batch
		biasgradients_per_batch_output = new Matrix[config.batchSize];
		biasgradients_per_batch_hidden = new Matrix[config.batchSize];
		weightgradients_per_batch_ho = new Matrix[config.batchSize];
		weightgradients_per_batch_ih = new Matrix[config.batchSize];
	}
	
	// creating a Accuracy class that will hold the stats needed to print the results of each epoch
	public static class Accuracy
	{
		boolean isCorrect;
		int outputclassification;
		int trueclassification;
		
		public Accuracy(boolean isCorrect, int outputclassification, int trueclassification)
		{
			this.isCorrect = isCorrect;
			this.outputclassification = outputclassification;
			this.trueclassification = trueclassification;
		}
		
		public int gettrueclassification()
		{
			return outputclassification;
		}
		
	}
	
	
	/* The Stochastic Gradient Descent (SGD) algorithm:
	   1. Randomize the order of the items in the training set.
	   2. Divide the training set into equal sized mini-batches, (e.g., ten items in each mini-batch)
	   3. Using backpropagation (see below) compute the weight gradients and bias gradients over the
	      first [next] mini-batch
	   4. After completing the mini-batch update the weights and biases as follows:
	      weightj k new = weightj k old – (learning_rate /size_of_mini-batch) * ∑ weight_gradientj k
	      biasj new = biasj old – (learning_rate /size_of_mini-batch) * ∑ bias_gradientj
	      where the summation is over the weight_gradients and bias_gradients returned from
	      backpropagating each input in the mini-batch.
	   5. If additional mini-batches remain, return to Step 3
	   6. If our stopping criteria have not been met (e.g. fixed # of times; accuracy), return to Step 1 */
	public void TrainwithSGD() 
	{
		// for each epoch
		for (int i=0; i < config.epochs; i++)
		{
			// clear stats per epoch
			AccuracyList.clear();
			System.out.println("\n=====Starting Epoch " + (i + 1) + "=====\n");

			// Randomize order but keeps classification with pixels
			randomizeArrays();
			
			//initialize weight and bias gradient matrices
			Matrix weightgradients_hidden_to_output = new Matrix(weights_hidden_to_output.rows, weights_hidden_to_output.columns);
			Matrix weightgradients_input_to_hidden = new Matrix(weights_input_to_hidden.rows, weights_input_to_hidden.columns);
			Matrix biasgradients_output = new Matrix(bias_output.rows, bias_output.columns);
			Matrix biasgradients_hidden = new Matrix(bias_hidden.rows, bias_output.columns);
			
			
			// loop through all 60k inputs. After every batch updating weights and biases 
			for (int j=0; j< inputArrays.length; j++)
			{
			//System.out.println("\n\nTRAINING CYCLE #"+j+"\n\n");
				// check if batch is done
				if (j > 0 && j % config.batchSize == 0 || j == inputArrays.length-1)
				{
					
					
					// update weights and biases
					updatebiases_hidden(biasgradients_per_batch_hidden, weights_input_to_hidden);
					updateweights_ih(weightgradients_per_batch_ih, weights_input_to_hidden);
					updatebiases_output(biasgradients_per_batch_output, bias_output);
					updateweights_ho(weightgradients_per_batch_ho, weights_hidden_to_output );
				}
				
				
				// for every image
				// feed forward
				
				
				// y = σ(z) where σ(z) is defined as 1 / (1 + e –z) and where z = (∑ (i=0 to (n-1)) Wi Xi) + b
				
				// converting pixel data and label set into Matrices using a method in my Matrix class
				Matrix input = Matrix.fromArray(inputArrays[j]);
				//input.printMatrix();
				
				// multiply the current weights by the input vector then add biases
				// then compute the sigmoid
				//weights_input_to_hidden.printMatrix();
				Matrix hidden = Matrix.multiply(weights_input_to_hidden, input);
				hidden.add(bias_hidden);
				hidden.sigmoid();
				//hidden.printMatrix();
				
				
		
				
				// Multiply hidden layer output by the weights between layer 1 and 2 and add biases
				// compute sigmoid
				Matrix output = Matrix.multiply(weights_hidden_to_output, hidden);
				output.add(bias_output);
				output.sigmoid();
				//output.printMatrix();
				
				// compare classification to correct output
				compareoutput(output, label_set[j]);
				
				// BACK-PROP OUTPUT LAYER
				biasgradients_output = backpropagateoutput(output, label_set[j]);
				biasgradients_per_batch_output[j % config.batchSize] = biasgradients_output;
				
				Matrix hidden_transposed  = Matrix.transpose(hidden);
				weightgradients_hidden_to_output = getweightgradients(biasgradients_output, hidden_transposed);
				weightgradients_per_batch_ho[j % config.batchSize] = weightgradients_hidden_to_output;
				
				//BACK-PROP HIDDEN LAYER
				
				biasgradients_hidden = backpropagatehidden(weights_hidden_to_output, hidden, biasgradients_output);
				biasgradients_per_batch_hidden[j % config.batchSize] = biasgradients_hidden;
				
				Matrix input_transposed = Matrix.transpose(input);
				weightgradients_input_to_hidden = getweightgradients(biasgradients_hidden, input_transposed);
				weightgradients_per_batch_ih[j % config.batchSize] = weightgradients_input_to_hidden;
				
			}
			printepochstats();
		}
		
	}
	
	public void randomizeArrays()
	{
		ThreadLocalRandom rndval = ThreadLocalRandom.current();
		// go through all 60k inputs
		for (int i = inputArrays.length-1; i > 0; i-- )
		{
			// get a random index
			int index = rndval.nextInt(i + 1);
			// swap input - create a temp array storing one line of input
			double[] temp = inputArrays[index];
			// put the first[next] index where the random index was
			inputArrays[index] = inputArrays[i];
			// replace the last[next] index of input array with the random line we got
			inputArrays[i] = temp;
			
			//swap classification
			temp = label_set[index];
			label_set[index] = label_set[i];
			label_set[i] = temp;
		}
	}
	
	/* The back propagation algorithm:
	[ Inputs: X (an input vector) and Y (the desired output vector) ]
	[ Outputs: gradient values for each weight and bias in the network ]
	1. Using the current weights and biases [which are initially random] along with an input vector X,
	compute the activations (outputs) of all neurons at all layers of the network. This is the “feed
	forward” pass.
	2. Using the computed output of the final layer together with the desired output vector Y,
	Compute the gradient of the error at the final level of the network and then move “backwards”
	through the network computing the error at each level, one level at a time. This is the
	“backwards pass”.
	3. Return as output the gradient values for each weight and bias in the network. */
	public Matrix backpropagateoutput(Matrix output,  double[] classification )
	{
		//move backwards
		// calculate bias gradient for last layer
		 // (a[i] - correct Classification) * a[i] * (1 - a[i])
		
		// converting classification from double array to column vector
		Matrix classificationvector = Matrix.fromArray(classification);
		// (output vector) - (classification vector)) or (a[i] - y[i])
		Matrix error = Matrix.subtract(output, classificationvector);
		
		// I have created a function in my matrix class that will calulate the derivative of sigmoid
		// (a * (1-a)) for each element in the Matrix so I make use of that here
		Matrix biasgradients_output = output.sigmoid_derivative();
		// now multiply the error times the derivative
		biasgradients_output.multiply(error);
		//biasgradients_output.printMatrix();
		return biasgradients_output;
		
		
	}
	
	
	public Matrix backpropagatehidden( Matrix weights, Matrix hidden, Matrix biasgradients_output )
	{
		// I have to transpose the weight matrix so that it will be a [15][10] (weights) matrix dot product with [10][1] (gradients)
		Matrix weights_transposed = Matrix.transpose(weights);
		// multiply weights by the bias gradients to get error
		Matrix error = Matrix.multiply(weights_transposed, biasgradients_output);
		// multiply error by sigmoid derivative (a) * (1-a)
		Matrix biasgradients_hidden = hidden.sigmoid_derivative();
		biasgradients_hidden.multiply(error);
		
		return biasgradients_hidden;
	}
	
	public Matrix getweightgradients(Matrix layeractivations, Matrix biasgradients)
	{
		// to get the weight gradients multiply the specified activations by the specidied bias gradients
		return Matrix.multiply(layeractivations, biasgradients);
	}
	
   /* update weights after each mini batch is done */
	public void updateweights_ho(Matrix[] weightgradients_per_batch, Matrix old_weights)
	{
		// for every old weight
		for(int i=0; i < old_weights.rows; i++)
		{
			
			// for the amount of columns in each row
			for (int j = 0; j < old_weights.columns; j++)
			{
				double gradientsum = 0;
				// go across all Matrices within the batch to get the value at i, j and add together to get the gradient sum
				for (int k=0; k < weightgradients_per_batch.length; k++)
				{
					gradientsum += weightgradients_per_batch[k].data[i][j];
				}
				
				//weightj k new = weightj k old – (learning_rate /size_of_training_data) * ∑ weight_gradientj k
				weights_hidden_to_output.data[i][j] =  old_weights.data[i][j] - ((config.learningRate/config.batchSize) * gradientsum);
			}
		}
		//weights_hidden_to_output.printMatrix();
	}
	
	public void updateweights_ih(Matrix[] weightgradients_per_batch, Matrix old_weights)
	{
		// for every old weight
		for(int i=0; i < old_weights.rows; i++)
		{
			
			// for the amount of columns in each row
			for (int j = 0; j < old_weights.columns; j++)
			{
				double gradientsum = 0;
				// go across Matrices to get the value at i, j
				for (int k=0; k < weightgradients_per_batch.length; k++)
				{
					gradientsum += weightgradients_per_batch[k].data[i][j];
				}
				
				//weightj k new = weightj k old – (learning_rate /_of_training_data) * ∑ weight_gradientj k
				weights_input_to_hidden.data[i][j] =  old_weights.data[i][j] - ((config.learningRate/config.batchSize) * gradientsum);
			}
		}
	}
	
	public void updatebiases_output(Matrix[] biasgradients_per_batch, Matrix old_biases)
	{
		// for every old bias
		for(int i=0; i < old_biases.rows; i++)
		{
			double gradientsum = 0;
			// for the amount of rows in bias gradient matrix
			for (int j = 0; j < old_biases.rows; j++)
			{
				// go across Matrices to get the values at k to sum up the gradients at that position across all images in the batch
				for (int k=0; k < biasgradients_per_batch.length; k++)
				{
					gradientsum += biasgradients_per_batch[k].data[j][0];
				}
			}
			
			// biasj new = biasj old – (learning_rate /size_of_mini-batch) * ∑ bias_gradientj
			List<Double> oldbiases = old_biases.toArray();
			bias_output.data[i][0] =  oldbiases.get(i) - ((config.learningRate/config.batchSize) * gradientsum);
		}
	}
	
	public void  updatebiases_hidden(Matrix[] biasgradients_per_batch, Matrix old_biases)
	{
		// for every old bias
		for(int i=0; i < old_biases.rows; i++)
		{
			double gradientsum = 0;
			// 
			for (int j = 0; j < old_biases.rows; j++)
			{
				// go across Matrices to get the values at k to sum up the gradients at that position across all images in the batch
				for (int k=0; k < biasgradients_per_batch.length; k++)
				{
					gradientsum += biasgradients_per_batch[k].data[j][0];
				}
			}
			List<Double> oldbiases = old_biases.toArray();
			
			// biasj new = biasj old – (learning_rate /size_of_mini-batch) * ∑ bias_gradientj
			bias_hidden.data[i][0] =  oldbiases.get(i) - ((config.learningRate/config.batchSize) * gradientsum);
		}
	}
	/* compare activation layer to correct output and tracks accuracy*/
	public void compareoutput(Matrix output, double[] classification)
	{	
		List<Double> outputList = output.toArray();
		Double[] outputArray = new Double[output.rows];
		outputArray = outputList.toArray(outputArray);
		
		// get the index of max activation value
		// function return an int array with maxIndex of the output activations at the 0 index and the maxIndex of the correctClassificationat the 1 index
		int[] maxindexes = getmaxelementindex(outputArray, classification);
		int maxOutput = maxindexes[0];
		int maxClassification = maxindexes[1];
		
		// add new accuracy to list to keep track of stats needed to print
		AccuracyList.add(new Accuracy((maxOutput == maxClassification), maxOutput, maxClassification));
	}
	
	public void printepochstats()
	{
		//  variables for overall tracking
		int totalCorrect = 0;
		int totalImgs = 0;
		
		// create Lists that will hold the amount of correct and incorrect classifications
		List<Accuracy> CorrectList = AccuracyList.stream().filter(c -> c.isCorrect).collect(Collectors.toList());
		List<Accuracy> IncorrectList = AccuracyList.stream().filter(c -> !(c.isCorrect)).collect(Collectors.toList());
		
		// go through each digit and display amount correct/total
		for (int i=0; i < 10; i++)
		{
			int amtCorrect = Collections.frequency(CorrectList.stream().map(Accuracy::gettrueclassification).collect(Collectors.toList()), i);
			int amtIncorrect = Collections.frequency(IncorrectList.stream().map(Accuracy::gettrueclassification).collect(Collectors.toList()), i);
			int totalDigit = amtCorrect + amtIncorrect;
			
			// get total correct for each digit
			totalCorrect += amtCorrect;
			totalImgs += totalDigit;
			
			// Display amount correct for each digit
			System.out.print(i + "=" + String.valueOf(amtCorrect) + "/" + String.valueOf(totalDigit) + "\t");
			
			
		}
		// display overall accuracy
		System.out.print("Accuracy = " + String.valueOf(totalCorrect) + "/" + String.valueOf(totalImgs) + " " + String.format("%.2f", ((double)totalCorrect / totalImgs) * 100) + "% \n");
	}
	
	public int[] getmaxelementindex(Double[] output, double[] classification )
	{
		// initialize matrix to store result
		int[] maxindexes = new int[2];
		// get the max index from the output activation
		int maxindex_output = 0;
		// go through each element in the array and if the current value is greater that the previous max then update the max to new value
		for (int i=0; i< output.length; i++)
		{
			if (output[i] > output[maxindex_output])
			{
				maxindex_output = i;
			}
		}
		maxindexes[0] = maxindex_output;
		
		// get the max from the correct classification
		int maxindex_classification = 0;
		for (int i=0; i< classification.length; i++)
		{
			if (classification[i] > output[maxindex_classification])
			{
				maxindex_classification = i;
			}
		}
		maxindexes[1] = maxindex_classification;
		
		return maxindexes;
	}
	
	/* function to convert the label digit value to one hot vector (double array)*/
	public double[] formatLabel(String labelval)
	{
		// parse string to integer and add appropriate amount of leading and trailing zeros
		int labeldigit = Integer.parseInt(labelval);
		String numLeadingZeros = new String(new char[labeldigit]).replace("\0", "0");
		String numTrailingZeros = new String(new char[(10-(labeldigit + 1))]).replace("\0","0");
		String onehotvectorstring = numLeadingZeros + "1" + numTrailingZeros;
		
		//convert from string to an array of doubles
		double[] onehotvector = Arrays.stream(onehotvectorstring.split(""))
                .mapToDouble(Double::parseDouble)
                .toArray();
		return onehotvector;
	}
	
	/* loads a pretrained network stored in the trained_network.csv file*/
	public void loadPretained() throws IOException
	{
		// go line by line
		try (BufferedReader br = new BufferedReader(new FileReader("trained_network.csv")))
		{
			String currline;
			int rowIndex = 0;
			while ((currline = br.readLine()) != null)
			{
				String[] vals = currline.split(",");
				// the first 30 rows are the weight values for input to hidden
				// I am going row by row in the file and inserting a double array at each row of the weight matrix
				// (rows 0 to 29) rowIndex < 30
				if(rowIndex < weights_input_to_hidden.rows)
				{
					// convert the current line string array into a double array and insert
					weights_input_to_hidden.data[rowIndex] = Arrays.stream(vals).mapToDouble(Double::valueOf).toArray();
				}
				// the next set of rows in the file are the weight values from the hidden to output
				// I am inserting a double array for each row of the weights matrix 
				// When I index into the weights from hidden to output Matrix I subtract the amount of rows of the previous weights in order
				// from the current File row index in order to start inserting into the weights matrix at index 0
				// (rows 30 to 39) rowIndex < 40
				else if(rowIndex < weights_input_to_hidden.rows + weights_hidden_to_output.rows)
				{ 
					weights_hidden_to_output.data[rowIndex - weights_input_to_hidden.rows] = Arrays.stream(vals).mapToDouble(Double::valueOf).toArray();
				}
				// the next set of rows are the bias values for the hidden layer
				// (rows 40 to 69) rowIndex < 70
				else if(rowIndex < weights_input_to_hidden.rows + weights_hidden_to_output.rows + bias_hidden.rows)
				{
					bias_hidden.data[rowIndex - weights_hidden_to_output.rows - weights_input_to_hidden.rows] = Arrays.stream(vals).mapToDouble(Double::valueOf).toArray();
				}
				//the next set of rows are the bias values for the ouput layer
				// (rows 70 to 79) rowIndex < 80
				else if (rowIndex < weights_input_to_hidden.rows + weights_hidden_to_output.rows + bias_hidden.rows )
				{
					bias_output.data[rowIndex - weights_hidden_to_output.rows - weights_input_to_hidden.rows - bias_hidden.rows] = Arrays.stream(vals).mapToDouble(Double::valueOf).toArray();
				}
				rowIndex++;
			}
		}
	}
	
	// Iterate over training data: make a forward pass and compare classification, print stats
	public void testNetwork_on_trainingData()
	{
		AccuracyList.clear();
		LoadData("C:\\Users\\reidt\\eclipse-workspace\\hdrnueralnetwork\\src\\hdrnueralnetwork\\mnist_train.csv");
		for (int i=0 ; i <inputArrays.length; i++)
		{
			Matrix input = Matrix.fromArray(inputArrays[i]);
			// forward pass
			Matrix hidden = Matrix.multiply(weights_input_to_hidden, input);
			hidden.add(bias_hidden);
			hidden.sigmoid();
			
			Matrix output = Matrix.multiply(weights_hidden_to_output, hidden);
			output.add(bias_output);
			output.sigmoid();
			
			compareoutput(output, label_set[i]);
		}
		printepochstats();
		
	}
	
	// Iterate over testing data: make forward pass and compare classification, print stats
	public void testNetwork_on_testingData()
	{
		AccuracyList.clear();
		LoadData_testing();
		for (int i=0 ; i <inputArrays_testing.length; i++)
		{
			Matrix input = Matrix.fromArray(inputArrays_testing[i]);
			// forward pass
			Matrix hidden = Matrix.multiply(weights_input_to_hidden, input);
			hidden.add(bias_hidden);
			hidden.sigmoid();
			
			Matrix output = Matrix.multiply(weights_hidden_to_output, hidden);
			output.add(bias_output);
			output.sigmoid();
			
			compareoutput(output, label_set_testing[i]);
		}
		printepochstats();
		
	}
	/* saving all weights and biases of the current network by writing all values to a specified csv file */
	public void saveNetwork() throws IOException
	{
		BufferedWriter br = new BufferedWriter(new FileWriter("trained_network.csv"));
		StringBuilder sb = new StringBuilder();
		
		// loop through the weights from input to hidden matrix and append each values into the string builder followed by a comma
		// for every row
		for (int i=0; i < weights_input_to_hidden.rows; i++)
		{
			// for every column
			for (int j=0; j < weights_input_to_hidden.columns; j++)
			{
				// add data to string builder
				sb.append(weights_input_to_hidden.data[i][j]);
				sb.append(",");
			}
			sb.append("\n");
		}
		
		// loop through the weights from hidden to ouput matrix and append each values into the string builder followed by a comma
		// for every row
		for (int i=0; i < weights_hidden_to_output.rows; i++)
		{
			// for every column
			for (int j=0; j < weights_hidden_to_output.columns; j++)
			{
				// add data to string builder
				sb.append(weights_hidden_to_output.data[i][j]);
				sb.append(",");
			}
			sb.append("\n");
		}
		// loop through the biases for hidden matrix and append each values into the string builder followed by a comma
		// for every row
		for (int i=0; i < bias_hidden.rows; i++)
		{
			// for every column
			for (int j=0; j < bias_hidden.columns; j++)
			{
				// add data to string builder
				sb.append(bias_hidden.data[i][j]);
				sb.append(",");
			}
			sb.append("\n");
		}
		// loop through the biases for ouput matrix and append each values into the string builder followed by a comma
		// for every row
		for (int i=0; i < bias_output.rows; i++)
		{
			// for every column
			for (int j=0; j < bias_output.columns; j++)
			{
				// add data to string builder
				sb.append(bias_output.data[i][j]);
				sb.append(",");
			}
			sb.append("\n");
		}
		br.write(sb.toString());
		br.close();
	}
	
	// Load in MNIST training data
	public void LoadData(String file)
	{
		try (BufferedReader br = new BufferedReader(new FileReader(file)))
		{
			int i=0;
			String currline;
			// go line by line
			while ((currline = br.readLine()) != null )
			{
				//System.out.println(i);
				// create string array to hold all values on line (785 vals)
				String[] vals = currline.split(",");
				// loop through to get the first element of each line (the label) and format
				for (int j = 0; j < vals.length; j++)
				{
					if (j == 0)
					{
					// grab the label (first element of line) then put in separate array
					// in one hot vector format
					label_set[i] = formatLabel(vals[0]);
					}
				}
				// now converting string array to double array and removing the label
				double[] inputArray = Arrays.stream(Arrays.copyOfRange(vals, 1, vals.length)).mapToDouble(Double::valueOf).toArray();
				// put each array of values into our inputArrays matrix with normalized input 0 to 1
				inputArrays[i] = DoubleStream.of(inputArray).map(p-> p/255).toArray();
				i++;
			}
		}
		catch (IOException ie)
		{
			System.out.println("Error loading data here");
			System.out.println(ie.getMessage());
		}
	}
	
	// Load in MNIIST Testing Data
	// to load the testing data in, seperate arrays were made since there only needs to be 10k rows, follow same process as previous loadData
	public void LoadData_testing( )
	{
		try (BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\reidt\\eclipse-workspace\\hdrnueralnetwork\\src\\hdrnueralnetwork\\mnist_test.csv")))
		{
			int i=0;
			String currline;
			// go line by line
			while ((currline = br.readLine()) != null )
			{
				// create string array to hold all values on line (785 vals)
				String[] vals = currline.split(",");
				// loop through to get the first element of each line (the label) and format
				for (int j = 0; j < vals.length; j++)
				{
					if (j == 0)
					{
					// grab the label (first element of line) then put in separate array
					// in one hot vector format
					label_set_testing[i] = formatLabel(vals[0]);
					}
				}
				// now converting string array to double array and removing the label
				double[] inputArray_testing = Arrays.stream(Arrays.copyOfRange(vals, 1, vals.length)).mapToDouble(Double::valueOf).toArray();
				// put each array of values into our inputArrays matrix with normalized input 0 to 1
				inputArrays_testing[i] = DoubleStream.of(inputArray_testing).map(p-> p/255).toArray();
				i++;
			}
		}
		catch (IOException ie)
		{
			System.out.println("Error loading data");
			System.out.println(ie.getMessage());
		}
	}
	
}

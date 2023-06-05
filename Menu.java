/* Reid Trisler
 * CWID: 10330107
 * 10/27/2022
 * CSC 475 Assignment #2 MNIST Handwritten Digit Neural Network in Java 
 * 
 * Fully Connected Feed Forward Neural Network trained using back propagation and stochastic gradient descent*/


package hdrnueralnetwork;


import java.io.IOException;
import java.util.Scanner;



public class Menu {
	public static void main(String[] args)
	{	
		Menu menu = new Menu();
		menu = menu.mainMenu(menu);
		System.out.println("Exiting Application");
	}
	
	private Menu mainMenu(Menu menu)
	{
		String[] menuOptions = 
		{	
			"[1] Train the network",
			"[2] Load a pre-trained network",
			"[0] Exit",
		};
		
		System.out.print("This is the Main Menu: \n");
		Scanner scanner = new Scanner(System.in);
		int option = 1;
		while(option != 0) 
		{
			printMenu(menuOptions);
			try 
			{
				option = scanner.nextInt();
				switch (option)
				{
					case 1: Network n = TrainNetwork(); Menu.subMenu(menu, n);break;
					case 2: Network n1 = LoadNetwork(); Menu.subMenu(menu, n1); break;
					case 0: System.exit(0);
				}
			}
			catch (Exception ex)
			{
				System.out.println(ex.getMessage());
				ex.printStackTrace();
				scanner.next();
			}
			
		}
		scanner.close();
		return menu;
		
	}
	
	public static Menu subMenu(Menu menu, Network n)
	{
		String[] submenuOptions = {
			   "[3] Display network accuracy on TRAINING data",
			   "[4] Display network accuracy on TESTING data",
			   "[5] Save the network state to file",
			   "[0] Exit",
		};
		System.out.print("Choose your option: \n");
		Scanner scanner = new Scanner(System.in);
		int option = 1;
		while(option != 0) 
		{
			printMenu(submenuOptions);
			try 
			{
				option = scanner.nextInt();
				switch (option)
				{
					case 3: TrainingAccuracy(n); Menu.subMenu(menu, n); break;
					case 4: TestingAccuracy(n); Menu.subMenu(menu, n);
					case 5: SaveState(n); Menu.subMenu(menu, n);
					case 0: System.exit(0);
				}
			}
			catch (Exception ex)
			{
				System.out.println(ex.getMessage());
				System.out.println("please choose a valid option");
				scanner.next();
			}
			
		}
		scanner.close();
		return menu;
	}
	
	
	public static void printMenu(String[] options)
	{
		for (String option: options)
		{
			System.out.println(option);
		}
	}
	
	private static Network TrainNetwork()
	{
		System.out.println("Training Network");
		NetworkConfig config = new NetworkConfig();
		config.epochs = 30;
		config.batchSize = 10;
		config.learningRate = 3.0;
		
		Network n = new Network(new int[] {784, 30, 10}, config);
		
		n.LoadData("C:\\Users\\reidt\\eclipse-workspace\\hdrnueralnetwork\\src\\hdrnueralnetwork\\mnist_train.csv");
		n.TrainwithSGD();
		return n;
	}
	private static Network LoadNetwork()
	{
		System.out.println("Loading Network...");
		NetworkConfig config = new NetworkConfig();
		config.epochs = 30;
		config.batchSize = 10;
		config.learningRate = 3.0;
		
		Network n = new Network(new int[] {784, 30, 10}, config);
		try {
			n.loadPretained();
		}
		catch (IOException ie)
		{
			ie.getMessage();
		}
		System.out.println("Network Loaded");
		return n;
	}
	private static void TrainingAccuracy(Network n)
	{
		System.out.println("Training Accuracy");
		n.testNetwork_on_trainingData();
	}
	private static void TestingAccuracy(Network n)
	{
		System.out.println("Testing Accuracy");
		n.testNetwork_on_testingData();
	}
	private static void SaveState(Network n)
	{
		System.out.println("Saving State...");
		try
		{
			n.saveNetwork();
		}
		catch (IOException ie)
		{
			ie.getMessage();
		}
		System.out.println("Network Saved");
	}
	
}

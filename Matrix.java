package hdrnueralnetwork;

import java.util.ArrayList;
import java.util.List;


/* Matrix class that is used to store the weights, biases, activations, gradients
 * and perform operations on the Matrices */



public class Matrix {
	//  initialiaze 2D array as well as rows and columns for the matrix
	double [][]data;
	int rows;
	int columns;
	
	// instantiate variables
	public Matrix(int rows, int columns)
	{
		data = new double[rows][columns];
		this.rows = rows;
		this.columns = columns;
		for (int i=0;i < rows; i++)
		{
			for (int j=0; j < columns; j++)
			{
				// initialize matrix with values between -1 and 1
				data[i][j]=Math.random()*2-1;
			}
		}
	}
	
	// returns column vector Matrix from a double array
	public static Matrix fromArray(double[] x)
    {
        Matrix temp = new Matrix(x.length,1);
        // for each item in the array place value into temp matrix at first[next] row
        for(int i =0;i<x.length;i++)
            temp.data[i][0]=x[i];
        return temp;
        
    }
	
	// puts contents of matrix into a list of doubles
	public List<Double> toArray()
	{
        List<Double> temp= new ArrayList<Double>()  ;
        // for each item in the Matrix add to list
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<columns;j++)
            {
                temp.add(data[i][j]);
            }
        }
        return temp;
   }
	
	// adding a transpose function for doing the back propagation matrix calculations
	public static Matrix transpose(Matrix a)
	{
		// temp matrix that reverses the rows and columns
		// Example: Matrix(2, 3) -> Matrix(3, 2)
        Matrix temp=new Matrix(a.columns,a.rows);
        // for each element in the Matrix add value into temp with the indices swapped
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.columns;j++)
            {
                temp.data[j][i]=a.data[i][j];
            }
        }
        return temp;
    }
	
	
	// implement matrix addition: I am overloading add function so that it handles scalar and matrix + matrix addition
	public void add(double scaler)
	{
		// for each row, go down column and add scaler to each element
		for (int i=0; i< rows; i++)
		{
			for (int j=0; j < columns; j++)
			{
				data[i][j] += scaler;
			}
		}
	}
	public void add(Matrix m)
	{
		// if the matrices match in size: for each row in the matrix go down the column and add the 
		// corresponding element from the other matrix
		if (rows != m.rows || columns != m.columns)
		{
			System.out.println("The matrix sizes do not matach");
			return;
		}
		for (int i=0; i< rows; i++)
		{
			for (int j=0; j < columns; j++)
			{
				this.data[i][j] += m.data[i][j];
			}
		}
	}
	
	//implement subtract 
	
	public static Matrix subtract(Matrix a, Matrix b) {
        Matrix temp=new Matrix(a.rows,a.columns);
        // subtract corresponding elements together and return result into temp array
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.columns;j++)
            {
                temp.data[i][j]=a.data[i][j]-b.data[i][j];
            }
        }
        return temp;
    }
	
	//implement matrix multiplication: I am overloading to handle scalar and matrix
	
	//multiply each value by scalar
	public void multiply(double scaler)
	{
		for (int i=0; i< rows; i++)
		{
			for (int j=0; j < columns; j++)
			{
				this.data[i][j] *= scaler;
			}
		}
	}
	// for multiplying by a column vector
	public void multiply(Matrix m)
	{
		for (int i=0; i< rows; i++)
		{
			for (int j=0; j < columns; j++)
			{
				this.data[i][j] *= m.data[i][j];
			}
		}
	}
	
	// multiplying two matrices of compatible lengths
	// Example Matrix(30 x 784) * Matrix(784, 1) = Matrix(30 x 1)
	public static Matrix multiply(Matrix a, Matrix b) {
		// temp array for result
        Matrix temp=new Matrix(a.rows,b.columns);
        // for each row in A Matrix
        for(int i=0;i<temp.rows;i++)
        {
        	// for each columns of B Matrix
            for(int j=0;j<temp.columns;j++)
            {
                double sum=0;
                // for each column(value) in A Matrix
                for(int k=0;k<a.columns;k++)
                {
                	// get the sum entire row of Matrix A after multiplying by correct element from Matrix b
                    sum+=a.data[i][k]*b.data[k][j];
                }
                temp.data[i][j]=sum;
            }
        }
        return temp;
    }
	
	// go through each value in the matrix and compute the sigmoid using that values
	// needed for activation matrices
	public void sigmoid() {
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<columns;j++)
                this.data[i][j] = 1/(1+Math.exp(-(this.data[i][j]))); 
        }
        
    }
	
	// go through all the data and compute (a) * (1-a)
	public Matrix sigmoid_derivative()
	{
		Matrix temp = new Matrix(rows, columns);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < columns; j++)
			{
				temp.data[i][j] = this.data[i][j] * (1-this.data[i][j]);
			}
		}
		return temp;
	}
	
	// for testing 
	public void printMatrix()
	{
		for(int i=0; i< this.rows; i++)
		{
			for(int j=0; j< this.columns; j++)
			{
				System.out.print("Row: " + i + " " + "Column: " + j + " " + this.data[i][j] + ", ");
			}
			System.out.println();
		}
	}
}

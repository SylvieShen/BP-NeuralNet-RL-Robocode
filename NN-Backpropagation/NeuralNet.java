import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;

public class NeuralNet implements NeuralNetInterface {
	static final int MAX_INPUTS_NUM = 20;
	static final int MAX_HIDDEN_NUM = 20;
	static final int MAX_OUTPUTS_NUM = 20;
	
    private int inputsNum;
    private int hiddenNum;
    private int outputsNum;
    private double learningRate;
    private double momentumTerm;
    private double argA;
    private double argB;
    private String activeType;

    // Weights from input layers to hidden layers
    private double[][] weight1 = new double[MAX_INPUTS_NUM][MAX_HIDDEN_NUM];
    // Weights form hidden layers to output layers
    private double[][] weight2 = new double[MAX_HIDDEN_NUM][MAX_OUTPUTS_NUM];
    // Weights difference between updated weights and previous one.
    private double[][] weightChange1 = new double[MAX_INPUTS_NUM][MAX_HIDDEN_NUM];
    private double[][] weightChange2 = new double[MAX_HIDDEN_NUM][MAX_OUTPUTS_NUM];

    private double[] inputsNeuron = new double[MAX_INPUTS_NUM ];
    private double[] hiddenNeuron = new double[MAX_HIDDEN_NUM];
    private double[] outputsNeuron = new double[MAX_OUTPUTS_NUM];
    // Value of delta in the hidden layer
    private double[] deltaHidden = new double[MAX_HIDDEN_NUM];
    // Value of delta in the output layer
    private double[] deltaOutput = new double[MAX_OUTPUTS_NUM];
    private double error = 0;

    public NeuralNet(int inputsNum, int hiddenNum, int outputsNum, 
    		         double learningRate, double momentumTerm,
                     double argA, double argB, String activeType) {
        this.inputsNum = inputsNum;
        this.hiddenNum = hiddenNum;
        this.outputsNum = outputsNum;
        this.learningRate = learningRate;
        this.momentumTerm = momentumTerm;
        this.argA = argA;
        this.argB = argB;
        this.activeType = activeType;
    }
  

    
    @Override
    public double sigmoid(double x) {
        return 2 / (1 + Math.exp(-x)) - 1;

    }

    
    @Override
    public double customSigmoid(double x) {
        return (argB - argA) / (1 + Math.exp(-x)) + argA;
    }

    
    
    @Override
    public void initializeWeights() {
        for (int i = 0; i < inputsNum + 1; i++) {
            // The last index represents the bias of input
            for (int h = 0; h < hiddenNum; h++) {
                weight1[i][h] = getRandomWeight(-0.5, 0.5);
                weightChange1[i][h] = 0.0;
            }
        }
        
        for (int h = 0; h < hiddenNum + 1; h++) {
            for (int j = 0; j < outputsNum; j++) {
                weight2[h][j] = getRandomWeight(-0.5, 0.5);
                weightChange2[h][j] = 0.0;
            }
        }
    }


    private double getRandomWeight(double minWeight, double maxWeight) {
        double random = new Random().nextDouble();
        return minWeight + (random * (maxWeight - minWeight));
    }


    @Override
    public void zeroWeights() {
        for (int i = 0; i < inputsNum + 1; i++) {
            for (int h = 0; h < hiddenNum; h++) {
                weight1[i][h] = 0.0;
            }
        }
        for (int h = 0; h < hiddenNum+1; h++) {
            for (int j = 0; j < outputsNum; j++) {
                weight2[h][j] = 0.0;
            }
        }
    }

    
    
    /*Construct neural net and do feed forward process, calculating the final output*/
    @Override
    public double outputFor(double[] X) {
        //Firstly, given the input X[], set up the neural net
        for(int i = 0;i < inputsNum; i++){
            inputsNeuron[i] = X[i];
        }
        //Add bias 
        inputsNeuron[inputsNum] = 1; 
        hiddenNeuron[hiddenNum] = 1;  

        //Compute hidden layer
        for(int h = 0; h < hiddenNum; h++){
        	hiddenNeuron[h] = 0;
            for(int i = 0;i < inputsNum + 1; i++){
                hiddenNeuron[h] += weight1[i][h] * inputsNeuron[i];
            }
            hiddenNeuron[h] = customSigmoid(hiddenNeuron[h]);
        }
        
        //Compute output layer
        for(int j = 0; j < outputsNum; j++){
        	outputsNeuron[j] = 0;
            for(int h = 0;h < hiddenNum + 1; h++){
                outputsNeuron[j] += weight2[h][j] * hiddenNeuron[h];
            }
            outputsNeuron[j] = customSigmoid(outputsNeuron[j]);
        }
        return outputsNeuron[0];  //Single output 
    }
    
    
    
    /*Backward process, computing the delta for each layer and then update weights*/
    private void updateWeight(double argValue){
        //Compute deltaOutput[] for output layer
        for(int j = 0; j < outputsNum; j++){
            if(activeType.equals("binary")) 
            deltaOutput[j] = (argValue - outputsNeuron[j]) * (1 - outputsNeuron[j]) * outputsNeuron[j];
            else if(activeType.equals("bipolar")) { 
             deltaOutput[j] = (argValue - outputsNeuron[j]) * 0.5 * (1 - outputsNeuron[j]) * (1 + outputsNeuron[j]);  
            }
        }
        //Update weights from output layer to hidden layer
        for(int j = 0; j < outputsNum; j++){
            for(int h = 0; h < hiddenNum + 1; h++){
                weight2[h][j] += momentumTerm * weightChange2[h][j] + learningRate * deltaOutput[j] * hiddenNeuron[h];
                weightChange2[h][j] = momentumTerm * weightChange2[h][j] + learningRate * deltaOutput[j] * hiddenNeuron[h];
            }
        }
        
        //Compute deltaHidden[] for hidden layer
        for(int h = 0; h < hiddenNum; h++){
            for(int j = 0; j < outputsNum; j++){
                deltaHidden[h] += deltaOutput[j] * weight2[h][j];
            }
            if(activeType.equals("binary"))
            deltaHidden[h] = (1 - hiddenNeuron[h]) * hiddenNeuron[h] * deltaHidden[h];
            else if(activeType.equals("bipolar")) {
            deltaHidden[h] = 0.5 * (1 - hiddenNeuron[h]) * (1 + hiddenNeuron[h]) * deltaHidden[h];
            }
        }      
        //Update weights from hidden layer to input layer
        for(int h = 0; h < hiddenNum; h++){
            for(int i = 0; i < inputsNum + 1; i++){
                weight1[i][h] += momentumTerm * weightChange1[i][h] + learningRate * deltaHidden[h] * inputsNeuron[i];
                weightChange1[i][h] = momentumTerm * weightChange1[i][h] + learningRate * deltaHidden[h] * inputsNeuron[i];
            }
        }
    }

  
    
    /*Train the network and return the error of each single neuron*/
    @Override
    public double train(double[] X, double argValue) {
    	double output;
    	try {
    		output = outputFor(X);
    		error = 0.5 * (argValue - output) * (argValue - output);
    		updateWeight(argValue);
    	}catch(Exception e) {
    		System.out.println(e);
    	}
        return error;
    }
        
    

   /*Save weights of a neural net*/
    @Override
    public void save(File argFile) {
        PrintStream saveWeight = null;
        try {
        	saveWeight = new PrintStream(new FileOutputStream(argFile));   	
        }catch(Exception e) {
        	System.out.println(e);
        }
        for(int h = 0; h < hiddenNum; h++){
        	for(int i = 0;i < inputsNum + 1; i++) {
        		saveWeight.println(weight1[i][h]);
        	}
        }
        for(int j = 0; j < outputsNum; j ++) {
        	for(int h = 0; h < hiddenNum+1; h++) {
        		saveWeight.println(weight2[h][j]);
        	}
        }
        saveWeight.close();  	
     }


    
    /*Load weights of a neural net from a file*/
    @Override
    public void load(String argFileName) throws IOException {
    	 FileInputStream weightFile = new FileInputStream(argFileName);
         BufferedReader weightReader = new BufferedReader(new InputStreamReader(weightFile));
         for(int h = 0; h < hiddenNum; h++){
         	for(int i = 0; i < inputsNum + 1; i++) {
         		weight1[i][h] = Double.valueOf(weightReader.readLine());
         	}
         }
         for(int j = 0; j < outputsNum; j++) {
         	for(int h = 0; h< hiddenNum + 1; h++) {
         		weight2[h][j] = Double.valueOf(weightReader.readLine());
         	}
         }
         weightReader.close();   
    }
}

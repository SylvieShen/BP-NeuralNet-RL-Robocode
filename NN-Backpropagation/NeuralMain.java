import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.LinkedList;

public class NeuralMain {

    public static void main(String[] args) {
    	/*initialize arguments*/
    	int inputsNum = 2;
    	int hiddenNum = 4;
    	int outputsNum = 1;
    	double learningRate = 0.2;
    	double momentumTerm = 0.9;
    	double argA = -1;   // For binary input, argA = 0 
    	double argB = 1;
    	String activeType = new String("bipolar");
	    
	    /*Binary test*/
//	    double inputs[][] = {{0, 0}, {0, 1},{ 1, 0}, {1, 1}};   
//	    double targets[] = {0, 1, 1, 0};
	    
	    /*Bipolar test*/
	    double inputs[][] = {{1, 1}, {1, -1},{ -1, 1}, {-1, -1}};  
	    double targets[] = {-1, 1, 1, -1};

	    double acceptError = 0.05;  
	  //String list to store error of each epoch
	    List<String> errorSave = new LinkedList<>();   
	  //Set training times
	    int trialsNum = 2000;  
	  //Average needed epochs after a number of trials
	    int epochsAvg = 0;

	    NeuralNet myNNet= new NeuralNet(inputsNum,hiddenNum,outputsNum,
	    		                        learningRate,momentumTerm,
	    		                        argA,argB,activeType);
	   
	    int epochMin = 10000;
	    int epochMax = 0;
	    for(int trial = 1;trial <= trialsNum; trial++) {
	    	//Initialize epochs for every trial
	    	int epochs = 0;  
	    	//Set error bigger than acceptError
	    	double error = 1.05;   
	    	myNNet.initializeWeights();
	    	while(error > acceptError) {
	    	error = 0;
	    	for (int i=0; i < hiddenNum; i++) {
	    		double[] inputNeuron = inputs[i];
	    		double argValue = targets[i];
			    error = error + myNNet.train(inputNeuron,argValue);			   
	    	}
	    	epochs++;
	    	if(trial == trialsNum / 2) {
				  System.out.println("epochs: "+epochs+" | "+"error: "+error);
				  errorSave.add(Double.toString(error));
			    }	    	
	        } 	    
	    	//Find min number of epochs
	    	if(epochs > epochMax) epochMax = epochs;  
	    	//Find max number of epochs
	    	if(epochs < epochMin) epochMin = epochs;  
	    	epochsAvg += epochs;
	        }   
	    
	    /*Save error data of the first trial to a text file*/
	    System.out.println("Above shows one example trial");
	    try {
		    Files.write(Paths.get("./errorSave.txt"), errorSave);
		    }catch(Exception e) {
		    	System.out.println(e);
		    }
	    
	    /*Calculate Average needed epochs*/
	    epochsAvg /= trialsNum;   
	    System.out.println("After "+ trialsNum + " trials, the average needed epochs is " + epochsAvg);
	    System.out.println("The max number of epochs:"+ epochMax);
	    System.out.println("The min number of epochs:"+ epochMin);
	}
}


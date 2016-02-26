package bayes.NaiveBayes;

import weka.core.Instances;


public class Main {
	
   public static void main(String[] args) throws Exception {

	   	Instances data; 	

		String path = args[0];
		data = Data.getData().cargar(path);
		System.out.println("The file " + path + " was loaded.");	
		
		Classifier c = new Classifier();
		
		c.naiveBayes(data);
   }
}


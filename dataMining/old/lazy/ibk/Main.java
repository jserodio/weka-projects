/*
 GOAL: Load data from .arff files, preprocess the data, train a model and assess it either using 10-fold cross-validation or hold-out
 
 Compile:
 javac DataMiningExample.java

 Run Interpret:
 java DataMiningExample
 
 HACER!!!
	- Hacer modular
	- El programa no puede tener dependencias con datos!
	- Generar un .jar y ejecutar desde la línea de comandos

 */
package lazy.ibk;

import weka.classifiers.Evaluation;
import weka.core.Instances;


public class Main {
	
    public static void main(String[] args) throws Exception {

    	Instances data; 	
    	
    	try {
    		String path = args[0];
	    	data = Data.getData().cargar(path);
			System.out.println("The file " + path + " was loaded.");
			
	    	Classifier c = new Classifier();
	    	Evaluation e_current = null;
	    	Evaluation e_max = null;
	    	Double max = 0.0;
	    	int maxk = 1, maxd = 1, maxw = 1;
	    	int d=1;
	    	int w=1;
	    	
	    	for (int k=2; k<=(data.numInstances()*0.9); k++) {
	    		System.out.println("==================================");
	    		System.out.println("===          " + k + "\t N  N     ");
	    		System.out.println("==================================");
	    		
	    	
	    		for (d=1; d<=3; d++) {
	    			for (w=1; w<=3; w++) {
	    				e_current = c.iBK(data, k, d, w);
	    				if (e_current.weightedFMeasure() > max){
	    					e_max = e_current;
	    					max = e_current.weightedFMeasure();
	    	    			maxk = k;
	    	    			maxd = d;
	    	    			maxw = w;
	    				}
	    			}
	    		}    		
	    	}
	    	
	    	System.out.println("Resultados del barrido de parametros...");	    	
	    	System.out.println("The k optimal parameter is:\t" + maxk);
	    	System.out.println("Maximum F-measure is:\t" + e_max.weightedFMeasure());
	    	System.out.println("Maximum d parameter is:\t" + maxd);
	    	System.out.println("Maximum w parameter is:\t" + maxw);
			System.out.println(e_max.toSummaryString());
			System.out.println(e_max.toClassDetailsString());
			System.out.println(e_max.toMatrixString());
	    	
			///////////////////////////////////////////////////////
			// Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
			///////////////////////////////////////////////////////
    	} catch(ArrayIndexOutOfBoundsException e){
    		e.printStackTrace();
    		System.out.println("");
    		System.out.println("Try this: java -jar knn.jar \"path/to folder/file.arff\"");
    		System.out.println("");
    	} catch(NullPointerException e){
    		e.printStackTrace();
    		System.out.println("");
    		System.out.println("Try this: java -jar knn.jar \"path/to folder/file.arff\"");
    		System.out.println("");
    	}
		
    }
}


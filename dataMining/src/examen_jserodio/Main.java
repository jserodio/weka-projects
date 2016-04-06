package examen_jserodio;

import weka.classifiers.Evaluation;
import weka.core.Instances;


public class Main {
	
    public static void main(String[] args) throws Exception {

    	Instances data; 
    	Instances newData;
    	int numInstancias;
    	int postNumInstancias;
    	
    	try {
    		String path = args[0]; // ruta fichero
    		//String path2 = args[1]; // ruta fichero guardar
    		
    		
	    	data = Data.getData().cargar(path);
			System.out.println("The file " + path + " was loaded.");
			// guardo el numero de instancias
			numInstancias = data.numInstances();
			System.out.println("Number of instances loaded: " + numInstancias);
			
			newData = Preprocess.getPreprocess().filtrar(data);
			// guardo el numero de instancias despues de filtrar
			postNumInstancias = newData.numInstances();
			System.out.println("Number of instances after filtering: " + postNumInstancias);
			
	    	Classifier c = new Classifier();
	    	
	    	Double maxFPR = 0.0;
	    	int maxNumTree = 0;
	    	Evaluation e_current = null;
	    	Evaluation e_max = null;
	    	
	    	// barrido
	    	for (int numTrees = 1; numTrees<=( 2^( newData.numAttributes() ) ); numTrees++ ){
	    		
	    		e_current = c.randomForest(newData, numTrees);
	    		
	    		if (e_current.falsePositiveRate(1)>maxFPR){
	    			// el nuevo maximo, sera el FPR actual
	    			maxFPR = e_current.falsePositiveRate(1);
	    			// el nuevo evaluador maximo, sera el actual
	    			e_max = e_current;
	    			maxNumTree = numTrees;
	    		}
	    	}
			
	    	System.out.println("Best numTree fount at: " + maxNumTree);
	    	System.out.println("Best FPR is: " + maxFPR);
	    	System.out.println("");
	    	System.out.println("=== Displaying summary FOR MAX FPR ===");
			System.out.println(e_max.toSummaryString());
			
			
			// evaluar con 10-fold cross validation
			c.evalCrossValidation(newData);
			System.out.println("=== Displaying summary FOR 10-FOLD CROSS VALIDATION ===");
			System.out.println(e_max.toSummaryString());

			// no me da tiempo
			c.evalBaseline(newData);
			
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


package bayes.NaiveBayes;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class Classifier {

	NaiveBayes estimador;

	public Classifier() {
		estimador= new NaiveBayes();//Naive Bayes
	}


	public void naiveBayes(Instances data) throws Exception{
		 				
		Evaluation evaluator;
								
		evaluator = new Evaluation(data);	
		
		// randomizar con 1, como en MORE OPTIONS -> Random seed for XVal / % Split = 1 en Weka GUI
		data.randomize(new Random(1));
		int trainSize = (int) Math.round(data.numInstances() * 0.66);
		int testSize = data.numInstances() - trainSize;
		
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		
		// Aprender con el 70% de las instancias (train)
		// creando el clasificador con el algoritmo Naive Bayes.
		estimador.buildClassifier(train);	
		
		// Dejar que prediga la clase estimada por el modelo para cada instancia del test
		// y así después podremos comparar la clase real y la estimada
		evaluator.evaluateModel(estimador, test);
		
		System.out.println(evaluator.toSummaryString());
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
					
	}

}

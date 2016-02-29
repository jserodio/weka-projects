package bayes.NaiveBayes;

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
		
		int trainSize = (int) Math.round(data.numInstances() * 0.7);
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
		
		double predictions[] = new double[test.numInstances()];
		
		// Recorrer todas las instancias
		for (int i = 0; i < test.numInstances(); i++) {
			predictions[i] = evaluator.evaluateModelOnceAndRecordPrediction(estimador, test.instance(i));
			//System.out.println(predictions[i]);
		}
					
	}

}

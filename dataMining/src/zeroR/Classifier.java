package zeroR;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

public class Classifier {

	ZeroR estimador;

	public Classifier() {
		estimador= new ZeroR();
	}


	public void zeroR(Instances data) throws Exception{
		 				
		Evaluation evaluator;
								
		evaluator = new Evaluation(data);	
		
		// Aprender con todos los datos, y luego hacer el Test con todos los datos.
		// creando el clasificador con el algoritmo zeroR.
		estimador.buildClassifier(data);	
		
		// Realizar el método de evaluación no-honesta (baseline), hacer el test con todos los datos.
		evaluator.evaluateModel(estimador, data);
		
		System.out.println(evaluator.toSummaryString());
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
					
	}

}

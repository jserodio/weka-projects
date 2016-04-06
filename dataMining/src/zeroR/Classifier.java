package zeroR;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

public class Classifier {

	CVParameterSelection estimador;

	public Classifier() throws Exception {
		estimador= new CVParameterSelection();
		Object[] o = new Object[4];
		o[0] = "K";
		o[1] = "1.0";
		o[2] = "10.0";
		o[3] = "10.0";
		estimador.setCVParameters(o);
		estimador.setSeed(1);
	}


	public void zeroR(Instances data) throws Exception{
		 		
		estimador.setClassifier(new IBk());
		
		Evaluation evaluator;
				
//		data.randomize(new Random(1));
		
		evaluator = new Evaluation(data);	
		
		// Aprender con todos los datos, y luego hacer el Test con todos los datos.
		// creando el clasificador con el algoritmo zeroR.
//		estimador.buildClassifier(data);	
		
		// Realizar el método de evaluación no-honesta (baseline), hacer el test con todos los datos.
		evaluator.crossValidateModel(estimador, data, 10, new Random(42));
		
		System.out.println(evaluator.toSummaryString());
//		System.out.println(evaluator.toClassDetailsString());
//		System.out.println(evaluator.toMatrixString());
					
	}

}

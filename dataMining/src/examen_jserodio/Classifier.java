package examen_jserodio;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class Classifier {
	
	RandomForest estimador;
	Evaluation evaluator;
	
	public Classifier() {
		estimador = new RandomForest();
	}


	public Evaluation randomForest(Instances data, int numTrees) throws Exception{
		
		estimador.setNumTrees(numTrees);				
		evaluator = new Evaluation(data);	
		Random rand = new Random(10);
		int folds = 4;
		
		// esquema de validacion cruzada cuadruple (4-fold cross validation)
		evaluator.crossValidateModel(estimador, data, folds, rand);	

		return evaluator;
	}
	
	public Evaluation evalCrossValidation(Instances data) throws Exception{
		
		evaluator = new Evaluation(data);	
		Random rand = new Random(11);
		int folds = 10;
		
		// esquema de validacion cruzada cuadruple (10-fold cross validation)
		evaluator.crossValidateModel(estimador, data, folds, rand);
		
		return evaluator;
	}

	public Evaluation evalBaseline(Instances data) throws Exception{
		
		// evaluando train=test=data
		evaluator.evaluateModel(estimador, data);
		
		return evaluator;
	}
	
}
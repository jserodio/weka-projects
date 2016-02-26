package bayes.NaiveBayes;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.ChebyshevDistance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;

public class Classifier {
	
	NaiveBayes estimador;
	ManhattanDistance manDistance;
	EuclideanDistance euDistance;
	ChebyshevDistance cheDistance;
	SelectedTag inverse;
	SelectedTag none;
	SelectedTag sim;
	
	public Classifier() {
		estimador= new NaiveBayes();//Naive Bayes


		
	}


	public Evaluation naiveBayes(Instances data) throws Exception{
		 				
				Evaluation evaluator;
				
					
					evaluator = new Evaluation(data);	
					
					int trainSize = (int) Math.round(data.numInstances() * 0.7);
					int testSize = data.numInstances() - trainSize;
					Instances train = new Instances(data, 0, trainSize);
					Instances test = new Instances(data, trainSize, testSize);
					

					
					
					//fmeasure = evaluator.weightedFMeasure();
					//accuracy = evaluator.pctCorrect();			
					
					/*
					 * Secret: http://weka.wikispaces.com/Generating+classifier+evaluation+output+manually
					 */

				return evaluator;
	}
	
}
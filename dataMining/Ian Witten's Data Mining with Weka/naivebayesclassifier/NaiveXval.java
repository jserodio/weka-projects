package naivebayesclassifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveXval {

	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("data/weather.nominal.arff");
		Instances data = source.getDataSet();
		
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random(1));
		
		System.out.println(classifier.toString());
		System.out.println("Time spent building the model: " + Math.round(eval.totalCost()) + " seconds.");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
	}

}

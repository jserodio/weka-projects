package naivebayesclassifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveHoldOut {

	public static void main(String[] args) throws Exception {
		
		
		
		DataSource source = new DataSource("data/weather.nominal.arff");
		Instances data = source.getDataSet();
		
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		data.randomize(new Random(1));
		int trainSize = (int) Math.round(data.numInstances() * 0.60);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(train);
		
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(classifier, test);
			
		System.out.println(eval.toSummaryString());
	}

}

package naiveBayesModel;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	static int missingValueCount, index;
	
	public static void main(String[] args) throws Exception {
		String output;
		DataSource source = new DataSource("data/balance-scale.arff");
		Instances data = source.getDataSet();
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		data.stream().forEach(instance -> {
			if (instance.hasMissingValue()) {
				missingValueCount++;
			}
			index = (int) instance.classValue();
			System.out.println(data.classAttribute().value(index));
			
		});
		
		output = "Data info: \n Missing Value count: " + missingValueCount;
		
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		
		// baseline
		eval.evaluateModel(classifier, data);
		output = output + "\n\n Baseline:\n" + classifier + "\n" + eval.toSummaryString();
		
		// 10-fold cv
		eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random(1));
		output = output + "\n\n10-Fold Cross Validation:\n" + eval.toSummaryString();
		
		// hold-out
		data.randomize(new Random(1));
		int trainSize = (int)Math.round(data.numInstances()*0.7);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		classifier.buildClassifier(train);
		eval = new Evaluation(train);
		eval.evaluateModel(classifier, test);
		
		output = output + "\n\nHold Out 70% for Training and 30% for Testing:\n" + classifier + "\n" + eval.toSummaryString();
		
		System.out.println(output);
		Files.write(Paths.get("data/output.txt"), output.getBytes());
	}

}

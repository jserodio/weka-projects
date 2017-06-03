package evaluation;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("data/weather.nominal.arff");
		Instances data = source.getDataSet();
		int numFolds = 10;
		Random random = new Random(1);
		String[] options;
		Evaluation evaluator;
		double fpr;
		double min = 999999.0;
		RandomForest classifier = null;
		RandomForest bestClassifier = null;
		Evaluation bestEvaluator = null;
		String output = "";
		
		if (data.classIndex()==-1) {
			data.setClassIndex(data.numAttributes()-1);
		}

		for (int trees = 1; trees<=100; trees++) {
			for (int seed = 1; seed <= 10; seed++){
				random = new Random(seed);
				classifier = new RandomForest();
				options = Utils.splitOptions("-I "+ trees +" -num-slots 1 -K 0 -S 1");
				classifier.setOptions(options);
				classifier.buildClassifier(data);
				
				evaluator = new Evaluation(data);
				evaluator.crossValidateModel(classifier, data, numFolds, random);
				
				fpr = evaluator.weightedFalsePositiveRate();
				if (fpr < min) {
					bestClassifier = classifier;
					bestEvaluator = evaluator;
					min = fpr;
					System.out.println("Min FPR found: " + min);
					System.out.println("Seed: " + seed);
					System.out.println("More Info: " + bestClassifier);
					output = "Min FPR found: " + min +"\n" +
							"Seed: " + seed + "\n" +
							 "More Info: " + bestClassifier;
							
				}
			}
		}
		
		System.out.println("> Sweeping parameters found this:");
		System.out.println(bestClassifier);
		System.out.println(bestEvaluator.toSummaryString());
		System.out.println("<<<<<<<<<<<<<<<<<<<<<<<");
		output = output + "> Sweeping parameters found this:" +"\n" +
				bestClassifier +"\n" +
				bestEvaluator.toSummaryString()+"\n" +
				"<<<<<<<<<<<<<<<<<<<<<<<";
		
		System.out.println("\n\n> Hold out");
		output = output + "\n\n> Hold out" + "\n";
		data.randomize(new Random(1));
		
		int trainSize = (int) Math.round(data.numInstances()*0.66);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		
		classifier = new RandomForest();
		options = Utils.splitOptions("-I 100 -num-slots 1 -K 0 -S 1");
		classifier.setOptions(options);
		
		classifier.buildClassifier(train);
		evaluator = new Evaluation(train);
		evaluator.evaluateModel(classifier, test);

		System.out.println(classifier);
		System.out.println(evaluator.toSummaryString());
		System.out.println("<<<<<<<<<<<<<<<<<<<<<<<");
		output = output + classifier + "\n" + evaluator.toSummaryString() + "\n<<<<<<<<<<<<<<<<<<<<<<<\n";
		
		// no-honesta
		System.out.println("\n\n> Test on the training set");
		output = output + "\n\n> Test on the training set\n";
		data.randomize(new Random(1));
		
		classifier = new RandomForest();
		options = Utils.splitOptions("-I 100 -num-slots 1 -K 0 -S 1");
		classifier.setOptions(options);
		
		classifier.buildClassifier(data);
		evaluator = new Evaluation(data);
		evaluator.evaluateModel(classifier, data);

		System.out.println(classifier);
		System.out.println(evaluator.toSummaryString());
		output = output +classifier + "\n" + evaluator.toSummaryString(); 
		
		Files.write(Paths.get("data/outputClassifier.txt"), output.getBytes());
	}

}

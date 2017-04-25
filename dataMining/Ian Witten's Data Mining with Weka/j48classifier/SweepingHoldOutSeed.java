package j48classifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class SweepingHoldOutSeed {

	public static void main(String[] args) throws Exception {
		
		// Load train data and set class
		DataSource source = new DataSource("data/segment-challenge.arff");
		Instances data = source.getDataSet();
		final double TOTAL = 9; 
		
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() - 1);
		}
		
		double max=0;
		int bestSeed = 0;
		Evaluation bestEval = null;
		J48 bestTree = null;
		double sum = 0;
		
		// loop 10 times.
		for (int i=1; i<=TOTAL; i++) {
			
			System.out.println("Trying seed: " + i);
			
			// reset data to randomize from the start
			data = source.getDataSet();
			if (data.classIndex() == -1) {
				data.setClassIndex(data.numAttributes() - 1);
			}
			// randomize data before split
			data.randomize(new Random(i));
			
			int trainSize = (int) Math.round(data.numInstances() * 0.9);
			int testSize = data.numInstances() - trainSize;
			Instances train = new Instances(data, 0, trainSize);
			Instances test = new Instances(data, trainSize, testSize);
			
			// Load and build the J48 classifier with default options
			J48 tree = new J48();
			String[] options = Utils.splitOptions("-C 0.25 -M 2");
			tree.setOptions(options);
			tree.buildClassifier(train);
			
			// Evaluate it on the separated test set.
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(tree, test);
			
			System.out.println(eval.toSummaryString());
			
			if (eval.pctCorrect() > max) {
				max = eval.pctCorrect();
				bestSeed = i;
				bestEval = eval;
				bestTree = tree;
			}
			
			sum += eval.pctCorrect();
		}
		
		System.out.println(bestTree);
		System.out.println("Luckiest seed was: " + bestSeed);
		System.out.println("Sample Mean: " + sum/(TOTAL*100) + ".");
		System.out.println(bestEval.toSummaryString());
	}

}

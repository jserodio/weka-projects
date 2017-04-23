package j48classifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class HoldOut {

	public static void main(String[] args) throws Exception {
		
		// Load train data and set class
		DataSource source = new DataSource("data/segment-challenge.arff");
		Instances data = source.getDataSet();
		
		// randomize data before split
		data.randomize(new Random(1));
		
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() - 1);
		}
		
		int trainSize = (int) Math.round(data.numInstances() * 0.66);
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
		
		System.out.println("Evaluating with Hold Out (66% for train, 33% for test).");
		System.out.println(eval.toSummaryString());
	}

}

package j48classifier;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class TrainVSTest {

	public static void main(String[] args) throws Exception {
		
		// Load train data and set class
		DataSource source = new DataSource("data/segment-challenge.arff");
		Instances data = source.getDataSet();
		
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() - 1);
		}
		
		// Load test data and set class
		DataSource sourceTest = new DataSource("data/segment-test.arff");
		Instances testData = sourceTest.getDataSet();
		
		if (testData.classIndex() == -1) {
			testData.setClassIndex(testData.numAttributes() - 1);
		}
		
		// Load and build the J48 classifier with default options
		J48 tree = new J48();
		String[] options = Utils.splitOptions("-C 0.25 -M 2");
		tree.setOptions(options);
		tree.buildClassifier(data);
		
		// Evaluate it on the separated test set.
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(tree, testData);
		
		System.out.println("Evaluating on the Test set.");
		System.out.println(eval.toSummaryString());
	}

}

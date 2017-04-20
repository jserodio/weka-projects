package j48classifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Basic {

	public static void main(String[] args) throws Exception {
		
		// get dataset
		DataSource source = new DataSource("data/glass.arff");
		Instances data = source.getDataSet();
		
		// set class attribute
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		// set options for J48/c4.5
		String[] options = weka.core.Utils.splitOptions("-C 0.25 -M 2");
		
		J48 tree = new J48();
		tree.setOptions(options);
		
		// build/train a classifier
		tree.buildClassifier(data);
		
		// test data with 10 fold cross validation
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(tree, data, 10, new Random(1));
		
		System.out.println("Prunned tree.");
		System.out.println("=== Stratified cross-validation ===");
		System.out.println(eval.toSummaryString("=== Summary ===\n", false));
		System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
		System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
	}

}

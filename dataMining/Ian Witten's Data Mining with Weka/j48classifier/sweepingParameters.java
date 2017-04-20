package j48classifier;

import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class sweepingParameters {

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
		
		Evaluation best = null;
		Evaluation eval = null;
		int bestNumber = 1;
		double max = 0;
		
		// at least get 2 leaves with 50% each
		for (int numInstances=1; numInstances<data.numInstances()/2; numInstances++) {
			tree.setMinNumObj(numInstances);
			eval = new Evaluation(data);
			eval.crossValidateModel(tree, data, 10, new Random(1));
			if (eval.pctCorrect()>max) {
				max = eval.pctCorrect();
				best = eval;
				bestNumber = numInstances;
			}
			System.out.println(numInstances + "...");
		}
		
		System.out.println("# Best number of instances per leaf: " + bestNumber);
		System.out.println(best.toSummaryString("=== Summary ===\n", false));
		System.out.println(best.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
		System.out.println(best.toMatrixString("=== Confusion Matrix ===\n"));
	
	}

}

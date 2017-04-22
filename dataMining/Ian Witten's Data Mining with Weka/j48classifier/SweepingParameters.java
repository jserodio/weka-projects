package j48classifier;

import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SweepingParameters {

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
		int bestInst = 1;
		float bestConf = 0;
		double max = 0;
		
		// declaring loop variables outside the loops
		int numInstances;
		float conf;
		
		for (conf=(float)0.1; conf<0.5; conf+=(float)0.1) {
			// at least get 2 leaves with 50% each
			for (numInstances=1; numInstances<data.numInstances()/2; numInstances++) {
				System.out.println(conf + "...");
				tree.setMinNumObj(numInstances);
				tree.setConfidenceFactor(conf);
				
				// build classifier
				tree.buildClassifier(data);
				
				eval = new Evaluation(data);
				eval.crossValidateModel(tree, data, 10, new Random(1));
				if (eval.pctCorrect()>max) {
					max = eval.pctCorrect();
					best = eval;
					bestInst = numInstances;
					bestConf = conf;
				}
				System.out.println(numInstances + "...");
			}
		}
		
		// build classifier again with the best
		tree.setMinNumObj(bestInst);
		tree.setConfidenceFactor(bestConf);
		tree.buildClassifier(data);
		
		System.out.println(tree);
		System.out.println("# Best MinNumObj: " + bestInst);
		System.out.println("# Best ConfidenceFactor: " + bestConf);
		System.out.println(best.toSummaryString("=== Summary ===\n", false));
		System.out.println(best.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
		System.out.println(best.toMatrixString("=== Confusion Matrix ===\n"));
	
	}

}

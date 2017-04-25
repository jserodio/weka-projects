package baseline;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ZeroRvsOthers {
	
	/**
	 * This is the ZeroR classifier, tested with the training set to get the baseline.
	 * J48, with default parameters. 
	 * Naive Bayes (has no parameters).
	 * IBk with default parameters.
	 * PART with default parameters. 
	 * These last 4 were tested with: Hold-Out method / percentage split (66% for training, and 33% for testing).
	 * 
	 */
	public static void main(String[] args) throws Exception {
		DataSource source;
		Instances data;
		double acZeroR, acJ48, acNaiveBayes, acIBk, acPART, max = 0;
		Classifier best = null;
		int trainSize, testSize;
		Instances train, test;
		
		/**
		 * DATA
		 */
		source = new DataSource("data/diabetes.arff");
		data = source.getDataSet();
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		/**
		 * ZeroR
		 */
		ZeroR classifier = new ZeroR();
		classifier.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(classifier, data);
		System.out.println(classifier);
		acZeroR = eval.pctCorrect();
		System.out.println(eval.toSummaryString());
//		System.out.println("Baseline accuracy is: " + Math.round(acZeroR) + "%");
		if (acZeroR >= max) {
			max = acZeroR;
			best = classifier;
		}
		
		/**
		 * Percentage split / hold out
		 */
		// randomize data before split
		data.randomize(new Random(1));
		trainSize = (int) Math.round(data.numInstances() * 0.66);
		testSize = (int) Math.round(data.numInstances() - trainSize);
		train = new Instances(data, 0, trainSize);
		test = new Instances(data, trainSize, testSize);
		
		/**
		 * J48/C4.5
		 */
		J48 tree = new J48();
		tree.buildClassifier(train);
		// you cannot reuse evaluation, otherwise instances will double
		eval = new Evaluation(train); // try commenting this line
		eval.evaluateModel(tree, test);
		acJ48 = eval.pctCorrect();
		System.out.println(eval.toSummaryString("== J48 ==", false));
//		System.out.println("C4.5 / J48 accuracy is: " + Math.round(acJ48) + "%");
		if (acJ48 >= max) {
			max = acJ48;
			best = tree;
		}

		/**
		 * Naive Bayes
		 */
		NaiveBayes bayes = new NaiveBayes();
		bayes.buildClassifier(train);
		// you cannot reuse evaluation, otherwise instances will double
		eval = new Evaluation(train); // try commenting this line
		eval.evaluateModel(bayes, test);
		acNaiveBayes = eval.pctCorrect();
		System.out.println(eval.toSummaryString("== Naive Bayes ==", false));
//		System.out.println("NaiveBayes accuracy is: " + Math.round(acNaiveBayes) + "%");
		if (acNaiveBayes >= max) {
			max = acNaiveBayes;
			best = bayes;
		}

		/**
		 * IBk / kNN
		 */
		IBk lazy = new IBk();
		lazy.buildClassifier(train);
		// you cannot reuse evaluation, otherwise instances will double
		eval = new Evaluation(train);
		eval.evaluateModel(lazy, test);
		acIBk = eval.pctCorrect();
		System.out.println(eval.toSummaryString("== IBk ==", false));
//		System.out.println("Instance Based / Nearest Neighbor accuracy is: " + Math.round(acIBk) + "%");
		if (acIBk >= max) {
			max = acIBk;
			best = lazy;
		}
		
		/**
		 * PART rules
		 */
		PART rule = new PART();
		rule.buildClassifier(train);
		// you cannot reuse evaluation, otherwise instances will double
		eval = new Evaluation(train);
		eval.evaluateModel(rule, test);
		acPART = eval.pctCorrect();
		System.out.println(eval.toSummaryString("== PART ==", false));
//		System.out.println("PART accuracy is: " + Math.round(acPART) + "%");
		if (acPART >= max) {
			max = acPART;
			best = rule;
		}
		
		System.out.println("The best result is: " + max + "%, the winner is: " + best.getClass().getSimpleName());
	}

}

package oneRclassifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

public class OneRxVal {

	public static void main(String[] args) throws Exception {
		
		DataSource source = new DataSource("data/weather.nominal.arff");
		Instances data = source.getDataSet();
		
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		OneR classifier = new OneR();
		String[] options = Utils.splitOptions("-B 6");
		classifier.setOptions(options);
		
		classifier.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random(1));
		
		System.out.println(classifier.toString());
		System.out.println("Time taken to build model: " + Math.round(eval.avgCost()) + " seconds");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
	}

}

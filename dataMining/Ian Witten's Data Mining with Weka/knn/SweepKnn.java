package knn;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.ChebyshevDistance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class SweepKnn {

	public static void main(String[] args) throws Exception {
		
		DataSource source = new DataSource("data/balance-scale.arff");
		Instances data = source.getDataSet();
		int d,w;

		ManhattanDistance manDistance = new ManhattanDistance();
		EuclideanDistance euDistance = new EuclideanDistance();
		ChebyshevDistance cheDistance = new ChebyshevDistance();
		SelectedTag inverse = new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING);
		SelectedTag none = new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING);
		SelectedTag sim = new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING);
		
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		IBk classifier = new IBk();
		classifier.buildClassifier(data);
		
		System.out.println("Classifier model (full-training set).");
		System.out.println(classifier.toString());
		
		for (d = 0; d<=4; d++) {
			for (w=0; w<=3; w++) {
				// options for IBk
				String[] options = Utils.splitOptions("-K "+(data.numInstances()-1)+" -W 0");
				classifier.setOptions(options);
				classifier.setCrossValidate(true);
				
				switch(d) {
				case 1:
					classifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(manDistance);
					break;
				case 2:
					classifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(euDistance);
					break;
				case 3:
					classifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(cheDistance);
					break;
				default:
		    		break;
				}
				
				switch(w) {
				case 1:
					classifier.setDistanceWeighting(inverse);
					break;
				case 2:
					classifier.setDistanceWeighting(none);
					break;
				case 3:
					classifier.setDistanceWeighting(sim);
					break;
				default:
		    		break;
				}

				classifier.buildClassifier(data);
				System.out.println(classifier);
				Evaluation eval = new Evaluation(data);
				eval.crossValidateModel(classifier, data, 10, new Random(1));
				System.out.println("Distance Function used in LinearSearch Algorithm: " + classifier.getNearestNeighbourSearchAlgorithm().getDistanceFunction().getClass().getSimpleName());
				System.out.println("Distance Weight used: " + classifier.getDistanceWeighting().getSelectedTag().getReadable());
				System.out.println(eval.toSummaryString());
				
				/**
				 * TODO: Comparar valor máximo, que favorezca el f-measure.
				 */
				
			}
		}
		
	}

}

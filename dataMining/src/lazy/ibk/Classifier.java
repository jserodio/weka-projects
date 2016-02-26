package lazy.ibk;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.ChebyshevDistance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;

public class Classifier {
	
	IBk estimador;
	ManhattanDistance manDistance;
	EuclideanDistance euDistance;
	ChebyshevDistance cheDistance;
	SelectedTag inverse;
	SelectedTag none;
	SelectedTag sim;
	
	public Classifier() {
		estimador = new IBk();
		manDistance = new ManhattanDistance();
		euDistance = new EuclideanDistance();
		cheDistance = new ChebyshevDistance();
		inverse = new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING);
		none = new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING);
		sim = new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING);
	}


	public Evaluation iBK(Instances data, int k, int d, int w) throws Exception{
		
				estimador.setKNN(k); 				
				Evaluation evaluator;
				
					
					switch(d) {
					case 1:
//			    		System.out.println("===     ManhattanDistance d=1     ");
						estimador.getNearestNeighbourSearchAlgorithm().setDistanceFunction(manDistance);
						break;
					case 2:
//			    		System.out.println("===     EuclideanDistance d=2     ");
						estimador.getNearestNeighbourSearchAlgorithm().setDistanceFunction(euDistance);
						break;
					case 3:
//			    		System.out.println("===     ChebyshevDistance d=3     ");
						estimador.getNearestNeighbourSearchAlgorithm().setDistanceFunction(cheDistance);
						break;
					default:
//						System.out.println(""); // starting
			    		break;
					}
					
					switch(w) {
					case 1:
//			    		System.out.println("===   1/distance(inverse) w=1     ");
						estimador.setDistanceWeighting(inverse);
						break;
					case 2:
//			    		System.out.println("===  	  none weigth w=2         ");
						estimador.setDistanceWeighting(none);
						break;
					case 3:
//			    		System.out.println("===  1-distance(similarity) w=3   ");
						estimador.setDistanceWeighting(sim);
						break;
					default:
//						System.out.println("Starting...");
			    		break;
					}
//					System.out.println("");
					
					evaluator = new Evaluation(data);	
					Random rand = new Random(1);
					int folds = 10;
					
					try{
						if (k>10 && d==1 && w==3){
							// tras los resultados obtenidos, se deduce que al combinar
	    					// manhattan con similarity y una k grande da problemas
						} else {
							evaluator.crossValidateModel(estimador, data, folds, rand);
						}
					}catch(ArrayIndexOutOfBoundsException e){
						// tras los resultados obtenidos, se deduce que al combinar
    					// manhattan con similarity y una k grande da problemas
						System.out.println("ERROR Manhattan distance is having problems with Weight-similarity in w parameter.");
					}
					
					//fmeasure = evaluator.weightedFMeasure();
					//accuracy = evaluator.pctCorrect();			
					
					/*
					 * Secret: http://weka.wikispaces.com/Generating+classifier+evaluation+output+manually
					 */

				return evaluator;
	}
	
}
package zeroR;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class Preprocess {

	private static Preprocess miPreprocess;
	
	private Preprocess () {

	}
	
	public static Preprocess getPreprocess () {
		if (miPreprocess == null){
			miPreprocess = new Preprocess();
		}
		return miPreprocess;
	}
	
	public Instances filtrar(Instances data) throws Exception{
//		/////////////////////////////////////////////////////////////		
//		// 2. FEATURE SUBSET SELECTION		
//		AttributeSelection filter = new AttributeSelection();
//		CfsSubsetEval eval = new CfsSubsetEval();
//		filter.setEvaluator(eval);
//		Ranker search = new Ranker();
//		filter.setSearch(search);
//		// 2.1 Get new data set with the attribute sub-set
//		Instances newData = Filter.useFilter(data, filter);

		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		filter.setEvaluator(eval);
		Ranker search = new Ranker();
		filter.setSearch(search);
		filter.setInputFormat(data);
		Instances newData = Filter.useFilter(data, filter);
		
		// Filtro normalizar atributos
//		Normalize filter = new Normalize();
//		filter.setInputFormat(data);
//		Instances newData = Filter.useFilter(data, filter);
		
		return newData;
	}

}

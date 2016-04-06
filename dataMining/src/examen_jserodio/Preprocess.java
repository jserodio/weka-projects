package examen_jserodio;

//import weka.attributeSelection.InfoGainAttributeEval;
//import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;
//import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;

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
		/////////////////////////////////////////////////////////////		
		// 2. FEATURE SUBSET SELECTION		
//		AttributeSelection filter= new AttributeSelection();
//		CfsSubsetEval eval = new CfsSubsetEval();
//		BestFirst search=new BestFirst();
//		filter.setEvaluator(eval);
//		filter.setSearch(search);
//		filter.setInputFormat(data);
//		// 2.1 Get new data set with the attribute sub-set
//		Instances newData = Filter.useFilter(data, filter);
		
		Instances newData;
		
		// Filtro interquartileRange
		InterquartileRange filterIR = new InterquartileRange();
		filterIR.setInputFormat(data);
		newData = Filter.useFilter(data, filterIR);
		
		// Filtro removeWithValues
		RemoveWithValues filterRV = new RemoveWithValues();
		filterRV.setInputFormat(newData);
		newData = Filter.useFilter(newData, filterRV);
		
		// Filtro remove
		Remove filterR = new Remove();
		filterR.setInputFormat(newData);
		newData = Filter.useFilter(newData, filterR);
		
		// Seleccion atributos
//		AttributeSelection filter = new AttributeSelection();
//		InfoGainAttributeEval eval = new InfoGainAttributeEval();
//		Ranker search = new Ranker();
//		search.setNumToSelect(10);
//		filter.setSearch(search);
//		filter.setInputFormat(newData);
//		newData = Filter.useFilter(newData, filter);
		
		return newData;
	}

}

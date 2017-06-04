package multifilter;

import java.nio.file.Files;
import java.nio.file.Paths;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class Main {

	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("data/breast-cancer.arff");
		Instances data = source.getDataSet();
		if (data.classIndex()==-1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		MultiFilter multi = new MultiFilter();
		// Filter 1
		AttributeSelection filter1 = new AttributeSelection();
		CfsSubsetEval evaluator1 = new CfsSubsetEval();
		BestFirst search1 = new BestFirst();
		filter1.setEvaluator(evaluator1);
		filter1.setSearch(search1);
		
		// Filter 2
		AttributeSelection filter2 = new AttributeSelection();
		InfoGainAttributeEval evaluator2 = new InfoGainAttributeEval();
		Ranker search2 = new Ranker();
		filter2.setEvaluator(evaluator2);
		filter2.setSearch(search2);
		
		// Filter 3
		RemoveWithValues filter3 = new RemoveWithValues();
		filter3.setAttributeIndex("2");
		filter3.setNominalIndices("2");

		Filter[] filtros = {filter1, filter2, filter3};
		multi.setFilters(filtros);
		multi.setInputFormat(data);
		
		Instances newData = Filter.useFilter(data, multi);
		
		Files.write(Paths.get("data/breast-cancer.multiFiltered.arff"), newData.toString().getBytes());
	}

}

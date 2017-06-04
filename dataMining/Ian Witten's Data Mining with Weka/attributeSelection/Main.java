package attributeSelection;

import java.nio.file.Files;
import java.nio.file.Paths;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class Main {

	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("data/weather.nominal.arff");
		Instances data = source.getDataSet();
		if (data.classIndex() == -1){
			data.setClassIndex(data.numAttributes()-1);
		}
		
		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval evaluator = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		
		filter.setEvaluator(evaluator);
		filter.setSearch(search);
		filter.setInputFormat(data);
		
		Instances newData = Filter.useFilter(data, filter);
		
		evaluator.buildEvaluator(data);
			
		System.out.println("Number of attributes before: " + data.numAttributes());
		for (int i=0; i<data.numAttributes(); i++)
			System.out.println(data.attribute(i).name());
		
		System.out.print("\n");
		System.out.println("Number of attributes after: " + newData.numAttributes());
		for (int i=0; i<newData.numAttributes(); i++)
			System.out.println(newData.attribute(i).name());
		
		Files.write(Paths.get("data/weather.nominal.filtered.arff"), newData.toString().getBytes());
	}

}

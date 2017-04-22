package filters;

import java.io.File;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;


/**
 * Just open Weather.nominal.arff dataset and
 * Remove instances where humidity is high.
 */
public class Basic {

	public static void main(String[] args) throws Exception {
		
		// load dataset
		DataSource source = new DataSource("data/weather.nominal.arff");
		Instances data = source.getDataSet();
		
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		
		System.out.println("Number of instances: " + data.numInstances());
		System.out.println("Aplying removeWithValues filter for humidity=high.");
		
		RemoveWithValues remV = new RemoveWithValues();
		
		String[] options = Utils.splitOptions("-C 3 -L 1");
		remV.setOptions(options);
		remV.setInputFormat(data);
		
		Instances filteredData = Filter.useFilter(data, remV);
		
		System.out.println("Number of instances: " + filteredData.numInstances());
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(filteredData);
		saver.setFile(new File("./data/filtered.arff"));
		//saver.setDestination(new File("./data/filtered.arff"));   // **not** necessary in 3.5.4 and later
		saver.writeBatch();
	}

}

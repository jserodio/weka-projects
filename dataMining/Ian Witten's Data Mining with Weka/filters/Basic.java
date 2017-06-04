package filters;

import java.nio.file.Files;
import java.nio.file.Paths;
import weka.core.Instances;
import weka.core.Utils;
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
		
		String[] options = Utils.splitOptions("-S 0.0 -C 3 -L 1");
		remV.setOptions(options);
		remV.setInputFormat(data);
		
		Instances filteredData = Filter.useFilter(data, remV);
		
		System.out.println("Number of instances: " + filteredData.numInstances());
		
		Files.write(Paths.get("data/weather.nominal.withoutHighHumidity.arff"), filteredData.toString().getBytes());
	}

}

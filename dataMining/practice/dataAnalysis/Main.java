package dataAnalysis;

import java.nio.file.Files;
import java.nio.file.Paths;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	static  int missingValues, brickfaceCount, skyCount, foliageCount, cementCount, windowCount, pathCount, grassCount, index;
	
	public static void main(String[] args) throws Exception {
		DataSource source;
		Instances data;
		String outputText;
		
		System.out.println("Loading... please wait!");
		//DataSource source = new DataSource(args[0]);
		source = new DataSource("data/segment-challenge.arff");
		data = source.getDataSet();
		if (data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		System.out.println("Done!");
		System.out.println("Analysing the data...");
		data.stream().forEach(instance -> {
					if (instance.hasMissingValue()) {
						missingValues++;
					}
					index = (int) instance.value(instance.numAttributes()-1);
					switch (instance.classAttribute().value(index)) {
						case "brickface":		brickfaceCount++;		break;
						case "sky":				skyCount++;				break;
						case "foliage":		foliageCount++;			break;
						case "cement":		cementCount++;			break;
						case "window":		windowCount++;		break;
						case "path":				pathCount++;				break;
						case "grass":			grassCount++;				break;
					}
		});
		
		outputText =
				"\n\n" 
				//"Number of attributes: " + data.numAttributes() + '\n'
				//+ "Number of instances: " + data.numInstances() + '\n'
				+ "Missing values count: " + missingValues + '\n'
				+	"Instances with 'brickface' class value: " + brickfaceCount + '\n'
				+	"Instances with 'sky' class value: " + skyCount + '\n'
				+	"Instances with 'foliage' class value: " + foliageCount + '\n'
				+	"Instances with 'cement' class value: " + cementCount + '\n'
				+	"Instances with 'window' class value: " + windowCount + '\n'
				+	"Instances with 'path' class value: " + pathCount + '\n'
				+	"Instances with 'grass' class value: " + grassCount + "\n\n"
				+	"Summary " + data.toSummaryString() + '\n';
		
		System.out.println(outputText);
		//Files.write(Paths.get(args[1]), outputText.getBytes());
		Files.write(Paths.get("data/output.txt"), outputText.getBytes());
	}
	
}
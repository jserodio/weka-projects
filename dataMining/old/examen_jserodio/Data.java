package examen_jserodio;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;

public class Data {
	
	private static Data miData;
	
	private Data (){ }
	
	public static Data getData (){
		if ( miData == null ) {
			miData = new Data();
		}
		return miData;
	}

	public Instances cargar(String path){
		FileReader fi=null;

		try {
		fi= new FileReader(path); 
		} catch (FileNotFoundException e) {
		System.out.println("ERROR: File not found "+path);
		}

		Instances data=null;
		try {
		data = new Instances(fi);
		} catch (IOException e) {
		System.out.println("ERROR: Check data inside the file: "+path);
		} catch (NullPointerException e) {
			System.out.println("ERROR: File is missing.");
		}

		try {
		fi.close();
		} catch (IOException e) {
			System.out.println("ERROR: Closing the file.");
		}

		data.randomize(new Random(42));
		

		if (data.attribute("class") != null){
			data.setClassIndex(data.attribute("class").index());
		} else {
			data.setClassIndex(data.numAttributes()-1);
		}

		return data;
	}
	
	// guardar fichero NO ME HA DADO TIEMPO
	// bufferedWritter (new fileWritter( new File(path2)))
	// write
	// flush
	// close
	
}

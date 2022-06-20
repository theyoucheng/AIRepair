import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class F1SelectionHarmonic {

	public static void main(String[] args) {
		
		String resultsPath = "/Users/experiments/nnrepair/cifar_adv_results";
		
		String subject = "CIFAR_LAST_LAYER_Eps0_01_ExpA";
		
		String f1PathAdversarialTrainingFile = resultsPath + "/" + subject + "_ADV_TRAINING_prec_f1.csv";
		String f1PathTrainingFile = resultsPath + "/" + subject + "_TRAINING_prec_f1.csv";
		
		
		double[] adv_train_f1_values = new double[10];
		double[] adv_train_f1_values_original  = new double[10];
		
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(f1PathAdversarialTrainingFile));
			String line = reader.readLine();
			while (line != null) {
				if (line.startsWith("f1Experts=[")) {
					System.out.println(line);
					System.out.println();
				}
				if (line.startsWith("f1_values=[")) {
					String[] values = line.substring(11, line.length()-1).split(",");
					for (int i=0; i<10; i++) {
						adv_train_f1_values[i] = Double.valueOf(values[i].trim());
					}
					System.out.println(line);
				}
				if (line.startsWith("f1_values_original=[")) {
					String[] values = line.substring(20, line.length()-1).split(",");
					for (int i=0; i<10; i++) {
						adv_train_f1_values_original[i] = Double.valueOf(values[i].trim());
					}
					System.out.println(line);
				}
				line = reader.readLine();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		double[] train_f1_values = new double[10];
		double[] train_f1_values_original  = new double[10];
		
		try {
			reader = new BufferedReader(new FileReader(f1PathTrainingFile));
			String line = reader.readLine();
			while (line != null) {
				if (line.startsWith("f1_values=[")) {
					String[] values = line.substring(11, line.length()-1).split(",");
					for (int i=0; i<10; i++) {
						train_f1_values[i] = Double.valueOf(values[i].trim());
					}
				}
				if (line.startsWith("f1_values_original=[")) {
					String[] values = line.substring(20, line.length()-1).split(",");
					for (int i=0; i<10; i++) {
						train_f1_values_original[i] = Double.valueOf(values[i].trim());
					}
				}
				line = reader.readLine();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		

		/////////

		List<Integer> f1HarmonicExperts = new ArrayList<>();
		double[] harmomic_f1_values = new double[adv_train_f1_values.length];
		double[] harmomic_f1_values_original = new double[adv_train_f1_values.length];

		for (int i = 0; i < adv_train_f1_values.length; i++) {
			harmomic_f1_values[i] = 2 * (adv_train_f1_values[i] * train_f1_values[i])
					/ (adv_train_f1_values[i] + train_f1_values[i]);
			harmomic_f1_values_original[i] = 2 * (adv_train_f1_values_original[i] * train_f1_values_original[i])
					/ (adv_train_f1_values_original[i] + train_f1_values_original[i]);

			if (harmomic_f1_values[i] > harmomic_f1_values_original[i]) {
				f1HarmonicExperts.add(i);
			}
		}

		System.out.println();
		System.out.println("harmomic_f1_values=" + Arrays.toString(harmomic_f1_values));
		System.out.println("harmomic_f1_values_original=" + Arrays.toString(harmomic_f1_values_original));
		System.out.println();
		System.out.println("f1HarmonicExperts=" + Arrays.toString(f1HarmonicExperts.toArray()));

	}

}

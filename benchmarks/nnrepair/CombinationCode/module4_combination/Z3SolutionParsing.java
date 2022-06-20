import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * I/O Operations - reading weights from Z3
 */
public class Z3SolutionParsing {

	public static Object loadRepairedWeights_CIFAR10(String path, String solutionFileNamePrefix, int repairedLayerId,
			int[] expertIDs, int numberOfExperts) throws IOException {

		if (repairedLayerId == 0) {
			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO
		} else if (repairedLayerId == 2) {
			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO
		} else if (repairedLayerId == 5) {
			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO
		} else if (repairedLayerId == 7) {
			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO
		} else if (repairedLayerId == 11) {
			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO
		} else if (repairedLayerId == 13) {
			/*
			 * 10 slots for experts, 1 slot for full repair, 1 slot for average weights of
			 * first 10 slots
			 */
			double[][][] weight_delta = new double[numberOfExperts + 2][512][10];

			ArrayList<Integer> num0 = new ArrayList<Integer>();
			ArrayList<Integer> num1 = new ArrayList<Integer>();
			ArrayList<Double> num2 = new ArrayList<Double>();

			/* Read deltas for experts 0..9 */
			for (int expertId : expertIDs) {
				loadDeltasFromZ3File_last(path, solutionFileNamePrefix, expertId, num0, num1, num2);
				for (int i = 0; i < num0.size(); i++) {
					weight_delta[expertId][num0.get(i)][expertId] = num2.get(i);
					System.out.println(expertId + " : " + num0.get(i) + " : " + expertId + " -> "
							+ weight_delta[expertId][num0.get(i)][expertId]);
				}
			}

			/* Read deltas for full repair. */
//			loadDeltasFromZ3File_last(path, NUMBER_OF_EXPERTS, num0, num1, num2);
//			for (int i = 0; i < num0.size(); i++) {
//				weight_delta[10][num0.get(i)][num0.get(i)] = num2.get(i);
//				System.out.println(NUMBER_OF_EXPERTS + " : " + num0.get(i) + " : " + NUMBER_OF_EXPERTS + " -> "
//						+ weight_delta[NUMBER_OF_EXPERTS][num0.get(i)][NUMBER_OF_EXPERTS]);
//				
//			}

			/* Calculate average deltas for experts. */
			for (int i = 0; i < 10; i++) {
				for (int I = 0; I < 512; I++) {
					double sum = 0.0;
					for (int expertId : expertIDs) {
						sum += weight_delta[expertId][I][i];
					}
					weight_delta[numberOfExperts + 1][I][i] = sum / expertIDs.length;
				}
			}

			return weight_delta;
			
		} else {
			throw new RuntimeException("Layer " + repairedLayerId + " cannot be repaired!");
		}

	}

	public static Object loadRepairedWeights_MNIST0(String path, String solutionFileNamePrefix, int repairedLayerId,
			int[] expertIDs, int numberOfExperts) throws IOException {

		if (repairedLayerId == 0) {

			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO

		} else if (repairedLayerId == 2) {

			throw new RuntimeException("Layer " + repairedLayerId + " not supported yet!"); // TODO

		} else if (repairedLayerId == 6) {

			/*
			 * 10 slots for experts, 1 slot for full repair, 1 slot for average weights of
			 * first 10 slots
			 */
			double[][][] weight_delta = new double[numberOfExperts + 2][576][128];

			ArrayList<Integer> num0 = new ArrayList<Integer>();
			ArrayList<Integer> num1 = new ArrayList<Integer>();
			ArrayList<Double> num2 = new ArrayList<Double>();

			/* Read deltas for experts 0..9 */
			for (int expertId : expertIDs) {
				loadDeltasFromZ3File_inter(path, expertId, num0, num1, num2);
				for (int i = 0; i < num1.size(); i++) {
					weight_delta[expertId][num1.get(i)][num0.get(i)] = num2.get(i);
					System.out.println(expertId + " : " + num1.get(i) + " : " + num0.get(i) + " -> "
							+ weight_delta[expertId][num1.get(i)][num0.get(i)]);
				}
			}

			/* Read deltas for full repair. */
			loadDeltasFromZ3File_inter(path, numberOfExperts, num0, num1, num2);
			for (int i = 0; i < num1.size(); i++) {
				weight_delta[10][num1.get(i)][num0.get(i)] = num2.get(i);
				System.out.println(numberOfExperts + " : " + num1.get(i) + " : " + num0.get(i) + " -> "
						+ weight_delta[numberOfExperts][num1.get(i)][num0.get(i)]);
			}

			/* Calculate average deltas for experts. */
			for (int i = 0; i < 128; i++) {
				for (int I = 0; I < 576; I++) {
					double sum = 0.0;
					for (int expertId : expertIDs) {
						sum += weight_delta[expertId][I][i];
					}
					weight_delta[numberOfExperts + 1][I][i] = sum / expertIDs.length;
				}
			}

			return weight_delta;
		} else if (repairedLayerId == 8) {
			/*
			 * 10 slots for experts, 1 slot for full repair, 1 slot for average weights of
			 * first 10 slots
			 */
			double[][][] weight_delta = new double[numberOfExperts + 2][128][10];

			ArrayList<Integer> num0 = new ArrayList<Integer>();
			ArrayList<Integer> num1 = new ArrayList<Integer>();
			ArrayList<Double> num2 = new ArrayList<Double>();

			/* Read deltas for experts 0..9 */
			for (int expertId : expertIDs) {
				loadDeltasFromZ3File_last(path, solutionFileNamePrefix, expertId, num0, num1, num2);
				for (int i = 0; i < num0.size(); i++) {
					weight_delta[expertId][num0.get(i)][expertId] = num2.get(i);
					System.out.println(expertId + " : " + num0.get(i) + " : " + expertId + " -> "
							+ weight_delta[expertId][num0.get(i)][expertId]);
				}
			}

			/* Read deltas for full repair. */
//			loadDeltasFromZ3File_last(path, NUMBER_OF_EXPERTS, num0, num1, num2);
//			for (int i = 0; i < num0.size(); i++) {
//				weight_delta[10][num0.get(i)][num0.get(i)] = num2.get(i);
//				System.out.println(NUMBER_OF_EXPERTS + " : " + num0.get(i) + " : " + NUMBER_OF_EXPERTS + " -> "
//						+ weight_delta[NUMBER_OF_EXPERTS][num0.get(i)][NUMBER_OF_EXPERTS]);
//				
//			}

			/* Calculate average deltas for experts. */
			for (int i = 0; i < 10; i++) {
				for (int I = 0; I < 128; I++) {
					double sum = 0.0;
					for (int expertId : expertIDs) {
						sum += weight_delta[expertId][I][i];
					}
					weight_delta[numberOfExperts + 1][I][i] = sum / expertIDs.length;
				}
			}

			return weight_delta;

		} else {
			throw new RuntimeException("Layer " + repairedLayerId + " cannot be repaired!");
		}

	}

	public static void loadDeltasFromZ3File_last(String path, String solutionFileNamePrefix, int lab,
			ArrayList<Integer> num0, ArrayList<Integer> num1, ArrayList<Double> num2) {

		num0.clear();
		num1.clear();
		num2.clear();

		String line = "";
		Pattern p = Pattern.compile("[-+]?[0-9]*\\.?[0-9]+");
		Matcher m;

		String readfile;
		if (lab == 10) {
			readfile = path + "/full.txt";
		} else {
			readfile = path + "/" + solutionFileNamePrefix + lab + ".txt";
		}

		try (FileReader frread = new FileReader(readfile); BufferedReader brread = new BufferedReader(frread)) {

			brread.readLine();
			brread.readLine();

			while ((line = brread.readLine()) != null) {
				while (line.contains("define-fun y")) {
					line = brread.readLine();
					line = brread.readLine();
					if (line.chars().filter(ch -> ch == '.').count() == 1) {
						line = brread.readLine();
					}
				}
				String a = "";
				String b = "";
				Double c = 0.0;
				int countnumbers = 0;
				// System.out.println("=>"+line);
				if (line.contains("define")) {
					line = line.replaceAll("_", " ");
					String[] nums = line.replaceAll("[^0-9 ]", "").trim().split(" +");
					// System.out.println(nums[1]);
					num0.add(Integer.valueOf(nums[0]));
//					num1.add(Integer.valueOf(nums[1]));
				} else if (line.contains("/") && line.chars().filter(ch -> ch == '.').count() == 2) {
					m = p.matcher(line);
					while (m.find()) {
						countnumbers++;
						if (countnumbers == 1) {
							a = m.group();
						} else if (countnumbers == 2) {
							b = m.group();
						}
					}
					c = Double.valueOf(a) / Double.valueOf(b);
					if (line.contains("-")) {
						c = c * -1;
					}
					num2.add(c);
				} else if (line.contains("/") && line.chars().filter(ch -> ch == '.').count() == 1) {
					m = p.matcher(line);
					while (m.find()) {
						a = m.group();
					}
					if (line.contains("-")) {
						c = -1.0;
					} else {
						c = 1.0;
					}

					line = brread.readLine();
					m = p.matcher(line);
					while (m.find()) {
						b = m.group();
					}
					c = c * Double.valueOf(a) / Double.valueOf(b);

					num2.add(c);
				} else {
					m = p.matcher(line);
					while (m.find()) {
						c = Double.valueOf(m.group());
					}
					if (line.contains("-")) {
						c = c * -1;
					}
					num2.add(c);
				}
			}
		} catch (IOException e) {
			throw new RuntimeException("Error during z3 output parsing.", e);
		}
	}

	public static void loadDeltasFromZ3File_inter(String path, int lab, ArrayList<Integer> num0,
			ArrayList<Integer> num1, ArrayList<Double> num2) {

		num0.clear();
		num1.clear();
		num2.clear();

		String line;
		Pattern p = Pattern.compile("[-+]?[0-9]*\\.?[0-9]+");

		String readfile;
		if (lab == 10) {
			readfile = path + "/full.txt";
		} else {
//			readfile = path + "/lowquality_label" + lab + ".txt";
			readfile = path + "/label" + lab + ".txt";
		}

		try (FileReader frread = new FileReader(readfile); BufferedReader brread = new BufferedReader(frread)) {
			line = "";
			Matcher m;
			brread.readLine();
			brread.readLine();

			while ((line = brread.readLine()) != null) {
				String a = "";
				String b = "";
				Double c = 0.0;
				int countnumbers = 0;
				// System.out.println("=>"+line);
				if (line.contains("define")) {
					line = line.replaceAll("_", " ");
					String[] nums = line.replaceAll("[^0-9 ]", "").trim().split(" +");
					// System.out.println(nums[1]);
					num0.add(Integer.valueOf(nums[0]));
					num1.add(Integer.valueOf(nums[1]));
				} else if (line.contains("/") && line.chars().filter(ch -> ch == '.').count() == 2) {
					m = p.matcher(line);
					while (m.find()) {
						countnumbers++;
						if (countnumbers == 1) {
							a = m.group();
						} else if (countnumbers == 2) {
							b = m.group();
						}
					}
					c = Double.valueOf(a) / Double.valueOf(b);
					if (line.contains("-")) {
						c = c * -1;
					}
					num2.add(c);
				} else if (line.contains("/") && line.chars().filter(ch -> ch == '.').count() == 1) {
					m = p.matcher(line);
					while (m.find()) {
						a = m.group();
					}
					if (line.contains("-")) {
						c = -1.0;
					} else {
						c = 1.0;
					}
					line = brread.readLine();
					m = p.matcher(line);
					while (m.find()) {
						b = m.group();
					}

					c = c * Double.valueOf(a) / Double.valueOf(b);

					num2.add(c);
				} else {
					m = p.matcher(line);
					while (m.find()) {
						c = Double.valueOf(m.group());
					}
					if (line.contains("-")) {
						c = c * -1;
					}
					num2.add(c);
				}

			}
		} catch (IOException e) {
			throw new RuntimeException("Error during z3 output parsing.", e);
		}
	}

}

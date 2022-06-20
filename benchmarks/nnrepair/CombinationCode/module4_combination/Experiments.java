import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Experiment Execution - Testing
 */
public class Experiments {

	/**
	 * Enumeration for supported models.
	 */
	enum SUBJECT {

		POISONED_CIFAR_LAST_LAYER_ExpA_TEST("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpA",
				"solution", 13, "\\cifar_test_label_csv.txt", "\\cifar_test_csv.txt", true, new double[] {},
				new int[] {5}, new int[] {}, "C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpA",
				"solution", 13, "\\cifar_train_label_csv.txt", "\\cifar_train_csv.txt", true, new double[] {},
				new int[] {5}, new int[] {}, "C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TEST("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpA",
				"solution", 13, "\\poisoned_cifar_test_label_csv.txt", "\\poisoned_cifar_test_csv.txt", true,
				new double[] {5}, new int[] {}, new int[] {},
				"C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TRAINING("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned",
				"/last/ExpA", "solution", 13, "\\poisoned_cifar_train_label_csv.txt", "\\poisoned_cifar_train_csv.txt",
				true, new double[] {5}, new int[] {}, new int[] {},
				"C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),


		POISONED_CIFAR_LAST_LAYER_ExpB_TEST("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpB",
				"solution", 13, "\\cifar_test_label_csv.txt", "\\cifar_test_csv.txt", true, new double[] {},
				new int[] {5}, new int[] {}, "C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpB_TRAINING("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpB",
				"solution", 13, "\\cifar_train_label_csv.txt", "\\cifar_train_csv.txt", true, new double[] {},
				new int[] {5}, new int[] {}, "C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpB_POISONED_TEST("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpB",
				"solution", 13, "\\poisoned_cifar_test_label_csv.txt", "\\poisoned_cifar_test_csv.txt", true,
				new double[] {}, new int[] {5}, new int[] {},
				"C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpB_POISONED_TRAINING("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned",
				"/last/ExpB", "solution", 13, "\\poisoned_cifar_train_label_csv.txt", "\\poisoned_cifar_train_csv.txt",
				true, new double[] {}, new int[] {5}, new int[] {},
				"C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),


		POISONED_CIFAR_LAST_LAYER_ExpC_TEST("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpC",
				"solution", 13, "\\cifar_test_label_csv.txt", "\\cifar_test_csv.txt", true, new double[] {},
				new int[] {}, new int[] {}, "C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpC_TRAINING("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpC",
				"solution", 13, "\\cifar_train_label_csv.txt", "\\cifar_train_csv.txt", true, new double[] {},
				new int[] {}, new int[] {}, "C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpC_POISONED_TEST("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpC",
				"solution", 13, "\\poisoned_cifar_test_label_csv.txt", "\\poisoned_cifar_test_csv.txt", true,
				new double[] {}, new int[] {}, new int[] {},
				"C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpC_POISONED_TRAINING("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned",
				"/last/ExpC", "solution", 13, "\\poisoned_cifar_train_label_csv.txt", "\\poisoned_cifar_train_csv.txt",
				true, new double[] {}, new int[] {}, new int[] {},
				"C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpD_TEST("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpD",
				"solution", 13, "\\cifar_test_label_csv.txt", "\\cifar_test_csv.txt", true, new double[] {},
				new int[] {5}, new int[] {}, "C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpD_TRAINING("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpD",
				"solution", 13, "\\cifar_train_label_csv.txt", "\\cifar_train_csv.txt", true, new double[] {},
				new int[] {5}, new int[] {}, "C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpD_POISONED_TEST("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned", "\\last\\ExpD",
				"solution", 13, "\\poisoned_cifar_test_label_csv.txt", "\\poisoned_cifar_test_csv.txt", true,
				new double[] {}, new int[] {5}, new int[] {},
				"C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results"),

		POISONED_CIFAR_LAST_LAYER_ExpD_POISONED_TRAINING("C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned",
				"/last/ExpD", "solution", 13, "\\poisoned_cifar_train_label_csv.txt", "\\poisoned_cifar_train_csv.txt",
				true, new double[] {}, new int[] {5}, new int[] {},
				"C:\\Users\\mlast\\Desktop\\experiments\\cifar_poisoned_results");

		////////////////////////////////////////////////////////////////////////////////////////////////////

		private String projectPath;
		private String repairPath;
		private String solutionFileNamePrefix;
		private int repairedLayerId;
		String inputFilePath;
		String labelFilePath;
		boolean needsNormalization;
		double[] trainPrecision;
		int[] f1SelectedExperts;
		int[] f1HarmonicSelectedExperts;
		String outputPath;

		SUBJECT(String projectPath, String repairPath, String solutionFileNamePrefix, int repairedLayerId,
				String labelFilePath, String inputFilePath, boolean needsNormalization, double[] trainPrecision,
				int[] f1SelectedExperts, int[] f1HarmonicSelectedExperts, String outputPath) {
			this.projectPath = projectPath;
			this.repairPath = projectPath + repairPath;
			this.solutionFileNamePrefix = solutionFileNamePrefix;
			this.repairedLayerId = repairedLayerId;
			this.labelFilePath = projectPath + labelFilePath;
			this.inputFilePath = projectPath + inputFilePath;
			this.needsNormalization = needsNormalization;
			this.trainPrecision = trainPrecision;
			this.f1SelectedExperts = f1SelectedExperts;
			this.f1HarmonicSelectedExperts = f1HarmonicSelectedExperts;
			this.outputPath = outputPath;
		}

		public String getProjectPath() {
			return projectPath;
		}

		public String getRepairPath() {
			return repairPath;
		}

		public String getSolutionFileNamePrefix() {
			return solutionFileNamePrefix;
		}

		public int getRepairedLayerId() {
			return repairedLayerId;
		}

		public String getLabelFilePath() {
			return labelFilePath;
		}

		public boolean needsNormalization() {
			return needsNormalization;
		}

		public String getInputFilePath() {
			return inputFilePath;
		}

		public double[] getTrainPrecision() {
			return trainPrecision;
		}

		public int[] getF1SelectedExperts() {
			return f1SelectedExperts;
		}

		public int[] getF1HarmonicSelectedExperts() {
			return f1HarmonicSelectedExperts;
		}

		public String getOutputPath() {
			return outputPath;
		}

	}

	/*
	 * *****************************************************************************
	 * Utilities
	 * *****************************************************************************
	 */

	private static double round(double value, int places) {
		if (Double.isNaN(value)) {
			return value;
		}
		if (places < 0)
			throw new IllegalArgumentException();
		BigDecimal bd = BigDecimal.valueOf(value);
		bd = bd.setScale(places, RoundingMode.HALF_UP);
		return bd.doubleValue();

	}

	/*
	 * *****************************************************************************
	 * Experiment Definitions
	 * *****************************************************************************
	 */

	public static void runMNIST0Experiment(SUBJECT subject, ExpertCombination.COMBINATION_METHOD combMethod)
			throws NumberFormatException, IOException {
		runMNIST0Experiment(subject, combMethod, null, false, false);
	}

	public static void runMNIST0Experiment(SUBJECT subject, ExpertCombination.COMBINATION_METHOD combMethod,
			Integer stopAfter, boolean useF1Selection, boolean useF1HarmonicSelection)
					throws NumberFormatException, IOException {

		if (useF1Selection && useF1HarmonicSelection) {
			throw new RuntimeException("You can use both: f1 selection and f1-harmonic selection.");
		}

		int repairedLayerId = subject.getRepairedLayerId(); // {0 | 2 | 6 | 8}

		System.out.println("PATH:" + subject.getProjectPath());

		/* Prepare the experts. */
		int[] expertIDs;
		if (useF1Selection) {
			expertIDs = subject.getF1SelectedExperts();
		} else if (useF1HarmonicSelection) {
			expertIDs = subject.getF1HarmonicSelectedExperts();
		} else {
			expertIDs = new int[MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS];
			for (int i = 0; i < MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS; i++) {
				expertIDs[i] = i;
			}
		}

		MNIST0_InternalData data = new MNIST0_InternalData(subject.getProjectPath(), "weights0.txt", "weights2.txt",
				"weights6.txt", "weights8.txt", "biases0.txt", "biases2.txt", "biases6.txt", "biases8.txt");
		Object repaired_weight_deltas = Z3SolutionParsing.loadRepairedWeights_MNIST0(subject.getRepairPath(),
				subject.getSolutionFileNamePrefix(), repairedLayerId, expertIDs,
				MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);
		MNIST0_DNNt_Combined model = new MNIST0_DNNt_Combined(data, repaired_weight_deltas);

		/* Initialize analytics */
		Map<Object, Integer> passCounter = new HashMap<>();
		Map<Object, Integer> failCounter = new HashMap<>();
		Map<Object, Integer> targetedPassCounter = new HashMap<>();
		Map<Object, Integer> targetedFailCounter = new HashMap<>();
		Map<Object, Integer> TPCounter = new HashMap<>();
		Map<Object, Integer> TNCounter = new HashMap<>();
		Map<Object, Integer> FPCounter = new HashMap<>();
		Map<Object, Integer> FNCounter = new HashMap<>();
		for (ExpertCombination.COMBINATION_METHOD x : ExpertCombination.COMBINATION_METHOD.values()) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
		}
		for (int x = 0; x < MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS + 2; x++) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
			targetedPassCounter.put(x, 0);
			targetedFailCounter.put(x, 0);
			TPCounter.put(x, 0);
			TNCounter.put(x, 0);
			FPCounter.put(x, 0);
			FNCounter.put(x, 0);
			String id = "ORIG_L" + x;
			targetedPassCounter.put(id, 0);
			targetedFailCounter.put(id, 0);
			TPCounter.put(id, 0);
			TNCounter.put(id, 0);
			FPCounter.put(id, 0);
			FNCounter.put(id, 0);
		}

		/* Read correct labels. */
		File file = new File(subject.getLabelFilePath());
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (stopAfter != null && index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Read input files and execute model. */
		file = new File(subject.getInputFilePath());
		br = new BufferedReader(new FileReader(file));
		int count = 0;

		while ((st = br.readLine()) != null) {
			// System.out.println("INPUT:" + st);

			String[] values = st.split(",");
			double[][][] input = new double[28][28][1];
			index = 0;
			while (index < values.length) {
				for (int i = 0; i < 28; i++)
					for (int j = 0; j < 28; j++)
						for (int k = 0; k < 1; k++) {
							Double val = Double.valueOf(values[index]);
							index++;
							if (subject.needsNormalization()) {
								input[i][j][k] = (double) (val / 255.0);
							} else {
								input[i][j][k] = (double) val;
							}
						}
			}

			Map<Integer, double[]> result = model.run(input, repairedLayerId, expertIDs);

			int correctLabel = labels[count];

			// Extract original decision.
			int origLabel = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1)); /* ORIG */

			// Determine final decisions by experts.
			Map<ExpertCombination.COMBINATION_METHOD, Integer> results = ExpertCombination.combineExperts(combMethod,
					result, origLabel, subject.getTrainPrecision(), expertIDs, false,
					MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);

			// Print results and collect analytics.
			System.out.print(count + "; IDEAL: " + correctLabel + "; ");
			for (Entry<ExpertCombination.COMBINATION_METHOD, Integer> combinedResult : results.entrySet()) {
				ExpertCombination.COMBINATION_METHOD currentCombinationMethod = combinedResult.getKey();
				int label = combinedResult.getValue();

				boolean passed = (label == correctLabel);
				if (passed) {
					passCounter.put(currentCombinationMethod, passCounter.get(currentCombinationMethod) + 1);
				} else {
					failCounter.put(currentCombinationMethod, failCounter.get(currentCombinationMethod) + 1);
				}

				System.out.print(currentCombinationMethod + ": " + (passed ? "PASS" : "FAIL") + " " + label + "; ");
			}

			// Collect results for experts. Accuracy is only interesting for targeted
			// repair. Precision is wanted for all experts.
			for (int expertId : expertIDs) {
				int label = ExpertCombination.selectLabelWithMaxConfidence(result.get(expertId));
				boolean passed = (label == correctLabel);
				if (passed) {
					passCounter.put(expertId, passCounter.get(expertId) + 1);
					if (correctLabel == expertId) {
						TPCounter.put(expertId, TPCounter.get(expertId) + 1);

						targetedPassCounter.put(expertId, targetedPassCounter.get(expertId) + 1); // only for local
						// expert
						System.out.print("ExpertL" + expertId + ": " + "PASS" + " " + label + "; ");
					} else {
						TNCounter.put(expertId, TNCounter.get(expertId) + 1);
					}
				} else {
					failCounter.put(expertId, failCounter.get(expertId) + 1);
					if (correctLabel == expertId) {
						FNCounter.put(expertId, FNCounter.get(expertId) + 1);

						targetedFailCounter.put(expertId, targetedFailCounter.get(expertId) + 1); // only for local
						// expert
						System.out.print("ExpertL" + expertId + ": " + "FAIL" + " " + label + "; ");
					} else if (label == expertId) {
						FPCounter.put(expertId, FPCounter.get(expertId) + 1);
					} else {
						TNCounter.put(expertId, TNCounter.get(expertId) + 1);
					}
				}

			}

			for (int expertId : expertIDs) {
				// Also collect results for original model specific for labels.
				boolean passed = origLabel == correctLabel;
				String id = "ORIG_L" + expertId;
				if (passed) {
					if (correctLabel == expertId) {
						TPCounter.put(id, TPCounter.get(id) + 1);
						targetedPassCounter.put(id, targetedPassCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					} else {
						TNCounter.put(id, TNCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					}

				} else {
					if (correctLabel == expertId) {
						FNCounter.put(id, FNCounter.get(id) + 1);
						targetedFailCounter.put(id, targetedFailCounter.get(id) + 1);
						System.out.print(id + ": " + "FAIL" + " " + origLabel + "; ");
					} else if (origLabel == expertId) {
						FPCounter.put(id, FPCounter.get(id) + 1);
						System.out.print(id + ": " + "FAIL" + " " + origLabel + "; ");
					} else {
						TNCounter.put(id, TNCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					}
				}
			}

			System.out.println();
			count++;

			if (stopAfter != null && count == stopAfter) {
				break;
			}

		}

		br.close();

		StringBuilder outStringBuilder = new StringBuilder();

		// Calculate and print accuracy.
		System.out.println();
		System.out.println("COMBINATION;ACCURACY;PASS;FAIL;TAR-ACC;TAR-PASS;TAR-FAIL;TP;TN;FP;FN;PREC;RECALL;F1");
		outStringBuilder
		.append("COMBINATION;ACCURACY;PASS;FAIL;TAR-ACC;TAR-PASS;TAR-FAIL;TP;TN;FP;FN;PREC;RECALL;F1\n");
		if (combMethod.equals(ExpertCombination.COMBINATION_METHOD.ALL)) {
			for (ExpertCombination.COMBINATION_METHOD combinationMethod : ExpertCombination.COMBINATION_METHOD
					.values()) {
				if (combinationMethod.equals(ExpertCombination.COMBINATION_METHOD.ALL)) {
					continue;
				}
				int pass = passCounter.get(combinationMethod);
				int fail = failCounter.get(combinationMethod);
				double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

				System.out.println(combinationMethod + ";" + accuracy + ";" + pass + ";" + fail + ";;;;;;;;;;");
				outStringBuilder
				.append(combinationMethod + ";" + accuracy + ";" + pass + ";" + fail + ";;;;;;;;;;" + "\n");
			}
		} else {
			int pass = passCounter.get(combMethod);
			int fail = failCounter.get(combMethod);
			double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

			System.out.println(combMethod + ";" + accuracy + ";" + pass + ";" + fail);
			outStringBuilder.append(combMethod + ";" + accuracy + ";" + pass + ";" + fail);
		}

		double[] prec = new double[MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS];
		double[] f1_values = new double[MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS];
		double[] f1_values_original = new double[MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS];
		List<Integer> f1Experts = new ArrayList<>();
		StringBuilder bs = new StringBuilder();

		for (int expertId : expertIDs) {
			int pass = passCounter.get(expertId);
			int fail = failCounter.get(expertId);
			double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

			int targetedPass = targetedPassCounter.get(expertId);
			int targetedFail = targetedFailCounter.get(expertId);
			double targetedAccuracy = round((((double) targetedPass) / (targetedPass + targetedFail)) * 100.0, 2);

			int TP = TPCounter.get(expertId);
			int TN = TNCounter.get(expertId);
			int FP = FPCounter.get(expertId);
			int FN = FNCounter.get(expertId);
			double precision = ((double) TP) / (TP + FP);
			prec[expertId] = precision;

			double recall = ((double) TP) / (TP + FN);
			double f1 = 2 * precision * recall / (precision + recall);

			System.out.println("L" + expertId + ";" + accuracy + ";" + pass + ";" + fail + ";" + targetedAccuracy + ";"
					+ targetedPass + ";" + targetedFail + ";" + TP + ";" + TN + ";" + FP + ";" + FN + ";"
					+ round(precision * 100.0, 2) + ";" + round(recall * 100.0, 2) + ";" + round(f1 * 100.0, 2));
			outStringBuilder.append("L" + expertId + ";" + accuracy + ";" + pass + ";" + fail + ";" + targetedAccuracy
					+ ";" + targetedPass + ";" + targetedFail + ";" + TP + ";" + TN + ";" + FP + ";" + FN + ";"
					+ round(precision * 100.0, 2) + ";" + round(recall * 100.0, 2) + ";" + round(f1 * 100.0, 2) + "\n");

			String id = "ORIG_L" + expertId;

			int targetedPass_O = targetedPassCounter.get(id);
			int targetedFail_O = targetedFailCounter.get(id);
			double targetedAccuracy_O = round((((double) targetedPass_O) / (targetedPass_O + targetedFail_O)) * 100.0,
					2);

			int TP_O = TPCounter.get(id);
			int TN_O = TNCounter.get(id);
			int FP_O = FPCounter.get(id);
			int FN_O = FNCounter.get(id);

			double precision_O = ((double) TP_O) / (TP_O + FP_O);
			double recall_O = ((double) TP_O) / (TP_O + FN_O);
			double f1_O = 2 * precision_O * recall_O / (precision_O + recall_O);

			bs.append(id + ";;;;" + targetedAccuracy_O + ";" + targetedPass_O + ";" + targetedFail_O + ";" + TP_O + ";"
					+ TN_O + ";" + FP_O + ";" + FN_O + ";" + round(precision_O * 100.0, 2) + ";"
					+ round(recall_O * 100.0, 2) + ";" + round(f1_O * 100.0, 2) + "\n");

			if (f1 > f1_O) {
				f1Experts.add(expertId);
			}

			f1_values[expertId] = f1;
			f1_values_original[expertId] = f1_O;
		}

		System.out.println(bs.toString());
		outStringBuilder.append(bs.toString());

		System.out.println();
		outStringBuilder.append("\n");
		System.out.println("prec=" + Arrays.toString(prec));
		System.out.println("f1Experts=" + Arrays.toString(f1Experts.toArray()));
		System.out.println("f1_values=" + Arrays.toString(f1_values));
		System.out.println("f1_values_original=" + Arrays.toString(f1_values_original));

		if (subject.getOutputPath() != null) {
			BufferedWriter writer = new BufferedWriter(new FileWriter(subject.getOutputPath() + "/" + subject.toString()
			+ (useF1Selection ? "_f1" : "") + (useF1HarmonicSelection ? "_f1har" : "") + ".csv"));
			writer.write(outStringBuilder.toString());
			writer.close();

			writer = new BufferedWriter(new FileWriter(subject.getOutputPath() + "/" + subject.toString()
			+ (useF1Selection ? "_f1" : "") + (useF1HarmonicSelection ? "_f1har" : "") + "_prec_f1.csv"));
			writer.write("prec=" + Arrays.toString(prec) + "\n");
			writer.write("f1Experts=" + Arrays.toString(f1Experts.toArray()) + "\n");
			writer.write("f1_values=" + Arrays.toString(f1_values) + "\n");
			writer.write("f1_values_original=" + Arrays.toString(f1_values_original) + "\n");
			writer.close();
		}

	}

	public static void runMNIST0CombinationOverheadExperiment(SUBJECT subject, Integer stopAfter, int iterations,
			boolean useF1Selection) throws NumberFormatException, IOException {

		int repairedLayerId = subject.getRepairedLayerId();

		/* Prepare the experts. */
		int[] expertIDs;
		if (useF1Selection) {
			expertIDs = subject.getF1SelectedExperts();
		} else {
			expertIDs = new int[MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS];
			for (int i = 0; i < MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS; i++) {
				expertIDs[i] = i;
			}
		}

		System.out.println("PATH:" + subject.getProjectPath());

		MNIST0_InternalData data = new MNIST0_InternalData(subject.getProjectPath(), "weights0.txt", "weights2.txt",
				"weights6.txt", "weights8.txt", "biases0.txt", "biases2.txt", "biases6.txt", "biases8.txt");
		//		Object repaired_weight_deltas = Z3SolutionParsing.loadRepairedWeights_MNIST0(subject.getRepairPath(),
		//				subject.getSolutionFileNamePrefix(), repairedLayerId, expertIDs,
		//				MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);
		double[][][] repaired_weight_deltas = new double[10 + 2][576][128];
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 576; j++) {
				for (int k = 0; k < 128; k++) {
					repaired_weight_deltas[i][j][k] = 0.5;
				}
			}
		}
		MNIST0_DNNt_Combined model = new MNIST0_DNNt_Combined(data, repaired_weight_deltas);
		MNIST0_DNNt_Original origModel = new MNIST0_DNNt_Original(data);

		/* Initialize analytics */
		double accumulatedTimeOriginal = 0;
		double accumulatedTimeCombinedNetwork = 0;
		double accumulatedTimeNAIVE = 0;
		double accumulatedTimeNAIVETotal = 0;
		double accumulatedTimePREC = 0;
		double accumulatedTimePRECTotal = 0;
		double accumulatedTimeVOTES = 0;
		double accumulatedTimeVOTESTotal = 0;
		double accumulatedTimeCONF = 0;
		double accumulatedTimeCONFTotal = 0;
		double accumulatedTimePVC = 0;
		double accumulatedTimePVCTotal = 0;

		/* Read correct labels. */
		File file = new File(subject.getLabelFilePath());
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (stopAfter != null && index == stopAfter) {
				break;
			}
		}
		br.close();

		StringBuilder outStringBuilder = new StringBuilder();

		/* Read input files and execute model. */
		file = new File(subject.getInputFilePath());
		br = new BufferedReader(new FileReader(file));
		int count = 0;

		while ((st = br.readLine()) != null) {
			// System.out.println("INPUT:" + st);

			String[] values = st.split(",");
			double[][][] input = new double[28][28][1];
			index = 0;
			while (index < values.length) {
				for (int i = 0; i < 28; i++)
					for (int j = 0; j < 28; j++)
						for (int k = 0; k < 1; k++) {
							Double val = Double.valueOf(values[index]);
							index++;
							if (subject.needsNormalization()) {
								input[i][j][k] = (double) (val / 255.0);
							} else {
								input[i][j][k] = (double) val;
							}
						}
			}

			System.out.print(count);

			// Run original model.
			long startTimeOriginal = System.currentTimeMillis();
			for (int i = 0; i < iterations; i++) {
				origModel.run(input);
			}
			double timeOriginal = (System.currentTimeMillis() - startTimeOriginal) / (double) iterations;
			accumulatedTimeOriginal += timeOriginal;
			System.out.print("; ORIG=" + timeOriginal);

			Map<Integer, double[]> result = new HashMap<>();
			// Run combination.
			long startTimeCombinationNetwork = System.currentTimeMillis();
			for (int i = 0; i < iterations; i++) {
				result = model.run(input, repairedLayerId, expertIDs, true);
			}
			double timeCombinationNetwork = (System.currentTimeMillis() - startTimeCombinationNetwork)
					/ (double) iterations;
			accumulatedTimeCombinedNetwork += timeCombinationNetwork;

			// Combine NAIVE.
			long startTimeCombinationNAIVE = System.currentTimeMillis();
			for (int i = 0; i < iterations; i++) {
				int origLabelNAIVE = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
				List<Integer> expertClaimsNAIVE = ExpertCombination.collectExpertClaims(expertIDs, result);
				ExpertCombination.combineExpertsByNaive(expertClaimsNAIVE, origLabelNAIVE);
			}
			double timeNAIVE = (System.currentTimeMillis() - startTimeCombinationNAIVE) / (double) iterations;
			accumulatedTimeNAIVE += timeNAIVE;
			accumulatedTimeNAIVETotal += (timeNAIVE + timeCombinationNetwork);
			System.out.print("; NAIVE=" + timeNAIVE + "; NAIVETotal=" + (timeNAIVE + timeCombinationNetwork));

			//			// Combine PREC.
			//			long startTimeCombinationPREC = System.currentTimeMillis();
			//			int origLabelPREC = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
			//			List<Integer> expertClaimsPREC = ExpertCombination.collectExpertClaims(expertIDs, result);
			//			ExpertCombination.combineExpertsByPrecision(expertClaimsPREC, origLabelPREC, subject.getTrainPrecision());
			//			long timePREC = System.currentTimeMillis() - startTimeCombinationPREC;
			//			accumulatedTimePREC += timePREC;
			//			accumulatedTimePRECTotal += (timePREC + timeCombinationNetwork);
			//			System.out.print("; PREC=" + timePREC + "; PRECTotal=" + (timePREC + timeCombinationNetwork));
			//
			// Combine VOTES.
			long startTimeCombinationVOTES = System.currentTimeMillis();
			for (int i = 0; i < iterations; i++) {
				int origLabelVOTES = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
				List<Integer> expertClaimsVOTES = ExpertCombination.collectExpertClaims(expertIDs, result);
				ExpertCombination.combineExpertsByVotes(result, expertClaimsVOTES, origLabelVOTES, expertIDs,
						MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);
			}
			double timeVOTES = (System.currentTimeMillis() - startTimeCombinationVOTES) / (double) iterations;
			accumulatedTimeVOTES += timeVOTES;
			accumulatedTimeVOTESTotal += (timeVOTES + timeCombinationNetwork);
			System.out.print("; VOTES=" + timeVOTES + "; VOTESTotal=" + (timeVOTES + timeCombinationNetwork));

			// Combine CONF.
			long startTimeCombinationCONF = System.currentTimeMillis();
			for (int i = 0; i < iterations; i++) {
				int origLabelCONF = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
				List<Integer> expertClaimsCONF = ExpertCombination.collectExpertClaims(expertIDs, result);
				ExpertCombination.combineExpertsByConfidence(result, expertClaimsCONF, origLabelCONF);
			}
			double timeCONF = (System.currentTimeMillis() - startTimeCombinationCONF) / (double) iterations;
			accumulatedTimeCONF += timeCONF;
			accumulatedTimeCONFTotal += (timeCONF + timeCombinationNetwork);
			System.out.print("; CONF=" + timeCONF + "; CONFTotal=" + (timeCONF + timeCombinationNetwork));

			//			// Combine PVC.
			//			long startTimeCombinationPVC = System.currentTimeMillis();
			//			int origLabelPVC = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1));
			//			List<Integer> expertClaimsPVC = ExpertCombination.collectExpertClaims(expertIDs, result);
			//			ExpertCombination.combineExpertsByPVC(result, expertClaimsPVC, origLabelPVC, subject.getTrainPrecision(),
			//					expertIDs, MNIST0_DNNt_Combined.NUMBER_OF_EXPERTS);
			//			long timePVC = System.currentTimeMillis() - startTimeCombinationPVC;
			//			accumulatedTimePVC += timePVC;
			//			accumulatedTimePVCTotal += (timePVC + timeCombinationNetwork);
			//			System.out.print("; PVC=" + timePVC + "; PVCTotal=" + (timePVC + timeCombinationNetwork));

			System.out.println();
			count++;

			if (stopAfter != null && count == stopAfter) {
				break;
			}

		}

		br.close();

		// Calculate and print average times.
		System.out.println();
		outStringBuilder
		.append("Average execution times after " + count + " inputs with " + iterations + " iterations.; \n");
		outStringBuilder.append("\n");
		outStringBuilder.append("\n");
		outStringBuilder.append("SUBJECT;AVG_TIME(ms)" + "\n");
		outStringBuilder.append("ORIG;" + ((double) accumulatedTimeOriginal / count) + "\n");
		outStringBuilder.append(";" + "\n");
		outStringBuilder.append("COMBINED_NETWORK;" + ((double) accumulatedTimeCombinedNetwork / count) + "\n");
		outStringBuilder.append(";" + "\n");
		outStringBuilder.append("NAIVE;" + ((double) accumulatedTimeNAIVE / count) + "\n");
		//		outStringBuilder.append("PREC;" + ((double) accumulatedTimePREC / count) + "\n");
		outStringBuilder.append("VOTES;" + ((double) accumulatedTimeVOTES / count) + "\n");
		outStringBuilder.append("CONF;" + ((double) accumulatedTimeCONF / count) + "\n");
		//		outStringBuilder.append("PVC;" + ((double) accumulatedTimePVC / count) + "\n");
		outStringBuilder.append(";" + "\n");
		outStringBuilder.append("NAIVETotal;" + ((double) accumulatedTimeNAIVETotal / count) + "\n");
		//		outStringBuilder.append("PRECTotal;" + ((double) accumulatedTimePRECTotal / count) + "\n");
		outStringBuilder.append("VOTESTotal;" + ((double) accumulatedTimeVOTESTotal / count) + "\n");
		outStringBuilder.append("CONFTotal;" + ((double) accumulatedTimeCONFTotal / count) + "\n");
		//		outStringBuilder.append("PVCTotal;" + ((double) accumulatedTimePVCTotal / count) + "\n");
		outStringBuilder.append("\n" + "\n");
		System.out.println(outStringBuilder.toString());

		if (subject.getOutputPath() != null) {
			BufferedWriter writer = new BufferedWriter(new FileWriter(subject.getOutputPath() + "/" + subject.toString()
			+ (useF1Selection ? "_f1" : "") + "_overhead.csv"));
			writer.write(outStringBuilder.toString());
			writer.close();
		}

	}

	public static void runOriginalCIFARDNN(SUBJECT subject, Integer stopAfter)
			throws NumberFormatException, IOException {

		System.out.println("PATH:" + subject.getProjectPath());

		CIFAR10_InternalData data = new CIFAR10_InternalData(subject.getProjectPath(), "weights0.txt", "weights2.txt",
				"weights5.txt", "weights7.txt", "weights11.txt", "weights13.txt", "biases0.txt", "biases2.txt",
				"biases5.txt", "biases7.txt", "biases11.txt", "biases13.txt");
		CIFAR10_DNNt_Original model = new CIFAR10_DNNt_Original(data);

		/* Initialize analytics */
		int passCounter = 0;
		int failCounter = 0;

		/* Read correct labels. */
		File file = new File(subject.getLabelFilePath());
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (stopAfter != null && index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Read input files and execute model. */
		file = new File(subject.getInputFilePath());
		br = new BufferedReader(new FileReader(file));
		int count = 0;

		while ((st = br.readLine()) != null) {
			// System.out.println("INPUT:" + st);

			String[] values = st.split(",");
			double[][][] input = new double[32][32][3];
			index = 0;
			while (index < values.length) {
				for (int i = 0; i < 32; i++)
					for (int j = 0; j < 32; j++)
						for (int k = 0; k < 3; k++) {
							Double val = Double.valueOf(values[index]);
							index++;
							if (subject.needsNormalization()) {
								input[i][j][k] = (double) (val / 255.0);
							} else {
								input[i][j][k] = (double) val;
							}

						}
			}

			int origLabel = model.run(input);
			int correctLabel = labels[count];
			boolean passed = (origLabel == correctLabel);

			// Print results and collect analytics.
			System.out.println(count + "; IDEAL: " + correctLabel + "; ORIG: " + (passed ? "PASS" : "FAIL") + " "
					+ origLabel + "");
			if (passed) {
				passCounter++;
			} else {
				failCounter++;
			}

			count++;

			if (stopAfter != null && count == stopAfter) {
				break;
			}

		}

		br.close();

		// Calculate and print accuracy.
		System.out.println();
		System.out.println("COMBINATION;ACCURACY;PASS;FAIL");

		double accuracy = round((((double) passCounter) / (passCounter + failCounter)) * 100.0, 2);
		System.out.println("ORIG" + ";" + accuracy + ";" + passCounter + ";" + failCounter);
		System.out.println();

	}

	public static void runCIFAR10Experiment(SUBJECT subject, ExpertCombination.COMBINATION_METHOD combMethod)
			throws NumberFormatException, IOException {
		runCIFAR10Experiment(subject, combMethod, null, false, false);
	}

	public static void runCIFAR10Experiment(SUBJECT subject, ExpertCombination.COMBINATION_METHOD combMethod,
			Integer stopAfter, boolean useF1Selection, boolean useF1HarmonicSelection)
					throws NumberFormatException, IOException {

		if (useF1Selection && useF1HarmonicSelection) {
			throw new RuntimeException("You can use both: f1 selection and f1-harmonic selection.");
		}

		int repairedLayerId = subject.getRepairedLayerId(); // {0 | 2 | 6 | 8}

		/* Prepare the experts. */
		int[] expertIDs;
		if (useF1Selection) {
			expertIDs = subject.getF1SelectedExperts();
		} else if (useF1HarmonicSelection) {
			expertIDs = subject.getF1HarmonicSelectedExperts();
		} else {
			expertIDs = new int[CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS];
			for (int i = 0; i < CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS; i++) {
				expertIDs[i] = i;
			}
		}

		System.out.println("PATH:" + subject.getProjectPath());

		CIFAR10_InternalData data = new CIFAR10_InternalData(subject.getProjectPath(), "weights0.txt", "weights2.txt",
				"weights5.txt", "weights7.txt", "weights11.txt", "weights13.txt", "biases0.txt", "biases2.txt",
				"biases5.txt", "biases7.txt", "biases11.txt", "biases13.txt");
		Object repaired_weight_deltas = Z3SolutionParsing.loadRepairedWeights_CIFAR10(subject.getRepairPath(),
				subject.getSolutionFileNamePrefix(), repairedLayerId, expertIDs,
				CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS);
		CIFAR10_DNNt_Combined model = new CIFAR10_DNNt_Combined(data, repaired_weight_deltas);

		/* Initialize analytics */
		Map<Object, Integer> passCounter = new HashMap<>();
		Map<Object, Integer> failCounter = new HashMap<>();
		Map<Object, Integer> targetedPassCounter = new HashMap<>();
		Map<Object, Integer> targetedFailCounter = new HashMap<>();
		Map<Object, Integer> TPCounter = new HashMap<>();
		Map<Object, Integer> TNCounter = new HashMap<>();
		Map<Object, Integer> FPCounter = new HashMap<>();
		Map<Object, Integer> FNCounter = new HashMap<>();
		for (ExpertCombination.COMBINATION_METHOD x : ExpertCombination.COMBINATION_METHOD.values()) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
		}
		for (int x = 0; x < CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS + 2; x++) {
			passCounter.put(x, 0);
			failCounter.put(x, 0);
			targetedPassCounter.put(x, 0);
			targetedFailCounter.put(x, 0);
			TPCounter.put(x, 0);
			TNCounter.put(x, 0);
			FPCounter.put(x, 0);
			FNCounter.put(x, 0);
			String id = "ORIG_L" + x;
			targetedPassCounter.put(id, 0);
			targetedFailCounter.put(id, 0);
			TPCounter.put(id, 0);
			TNCounter.put(id, 0);
			FPCounter.put(id, 0);
			FNCounter.put(id, 0);
		}

		/* Read correct labels. */
		File file = new File(subject.getLabelFilePath());
		BufferedReader br = new BufferedReader(new FileReader(file));
		String st;
		Integer[] labels = new Integer[60000];
		int index = 0;
		while ((st = br.readLine()) != null) {
			labels[index] = Integer.valueOf(st);
			index++;
			if (stopAfter != null && index == stopAfter) {
				break;
			}
		}
		br.close();

		/* Read input files and execute model. */
		file = new File(subject.getInputFilePath());
		br = new BufferedReader(new FileReader(file));
		int count = 0;

		while ((st = br.readLine()) != null) {
			// System.out.println("INPUT:" + st);

			String[] values = st.split(",");
			double[][][] input = new double[32][32][3];
			index = 0;
			while (index < values.length) {
				for (int i = 0; i < 32; i++)
					for (int j = 0; j < 32; j++)
						for (int k = 0; k < 3; k++) {
							Double val = Double.valueOf(values[index]);
							index++;
							if (subject.needsNormalization()) {
								input[i][j][k] = (double) (val / 255.0);
							} else {
								input[i][j][k] = (double) val;
							}

						}
			}

			Map<Integer, double[]> result = model.run(input, repairedLayerId, expertIDs);

			int correctLabel = labels[count];

			// Extract original decision.
			int origLabel = ExpertCombination.selectLabelWithMaxConfidence(result.get(-1)); /* ORIG */

			// Determine final decisions by experts.
			Map<ExpertCombination.COMBINATION_METHOD, Integer> results = ExpertCombination.combineExperts(combMethod,
					result, origLabel, subject.getTrainPrecision(), expertIDs, false,
					CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS);

			// Print results and collect analytics.
			System.out.print(count + "; IDEAL: " + correctLabel + "; ");
			for (Entry<ExpertCombination.COMBINATION_METHOD, Integer> combinedResult : results.entrySet()) {
				ExpertCombination.COMBINATION_METHOD currentCombinationMethod = combinedResult.getKey();
				int label = combinedResult.getValue();

				boolean passed = (label == correctLabel);
				if (passed) {
					passCounter.put(currentCombinationMethod, passCounter.get(currentCombinationMethod) + 1);
				} else {
					failCounter.put(currentCombinationMethod, failCounter.get(currentCombinationMethod) + 1);
				}

				System.out.print(currentCombinationMethod + ": " + (passed ? "PASS" : "FAIL") + " " + label + "; ");
			}

			// Collect results for experts. Accuracy is only interesting for targeted
			// repair. Precision is wanted for all experts.
			for (int expertId : expertIDs) {
				int label = ExpertCombination.selectLabelWithMaxConfidence(result.get(expertId));
				boolean passed = (label == correctLabel);
				if (passed) {
					passCounter.put(expertId, passCounter.get(expertId) + 1);
					if (correctLabel == expertId) {
						TPCounter.put(expertId, TPCounter.get(expertId) + 1);

						targetedPassCounter.put(expertId, targetedPassCounter.get(expertId) + 1); // only for local
						// expert
						System.out.print("ExpertL" + expertId + ": " + "PASS" + " " + label + "; ");
					} else {
						TNCounter.put(expertId, TNCounter.get(expertId) + 1);
					}
				} else {
					failCounter.put(expertId, failCounter.get(expertId) + 1);
					if (correctLabel == expertId) {
						FNCounter.put(expertId, FNCounter.get(expertId) + 1);

						targetedFailCounter.put(expertId, targetedFailCounter.get(expertId) + 1); // only for local
						// expert
						System.out.print("ExpertL" + expertId + ": " + "FAIL" + " " + label + "; ");
					} else if (label == expertId) {
						FPCounter.put(expertId, FPCounter.get(expertId) + 1);
					} else {
						TNCounter.put(expertId, TNCounter.get(expertId) + 1);
					}
				}

			}

			for (int expertId : expertIDs) {
				// Also collect results for original model specific for labels.
				boolean passed = origLabel == correctLabel;
				String id = "ORIG_L" + expertId;
				if (passed) {
					if (correctLabel == expertId) {
						TPCounter.put(id, TPCounter.get(id) + 1);
						targetedPassCounter.put(id, targetedPassCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					} else {
						TNCounter.put(id, TNCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					}

				} else {
					if (correctLabel == expertId) {
						FNCounter.put(id, FNCounter.get(id) + 1);
						targetedFailCounter.put(id, targetedFailCounter.get(id) + 1);
						System.out.print(id + ": " + "FAIL" + " " + origLabel + "; ");
					} else if (origLabel == expertId) {
						FPCounter.put(id, FPCounter.get(id) + 1);
						System.out.print(id + ": " + "FAIL" + " " + origLabel + "; ");
					} else {
						TNCounter.put(id, TNCounter.get(id) + 1);
						System.out.print(id + ": " + "PASS" + " " + origLabel + "; ");
					}
				}
			}

			System.out.println();
			count++;

			if (stopAfter != null && count == stopAfter) {
				break;
			}

		}

		br.close();

		StringBuilder outStringBuilder = new StringBuilder();

		// Calculate and print accuracy.
		System.out.println();
		System.out.println("COMBINATION;ACCURACY;PASS;FAIL;TAR-ACC;TAR-PASS;TAR-FAIL;TP;TN;FP;FN;PREC;RECALL;F1");
		outStringBuilder
		.append("COMBINATION;ACCURACY;PASS;FAIL;TAR-ACC;TAR-PASS;TAR-FAIL;TP;TN;FP;FN;PREC;RECALL;F1\n");
		if (combMethod.equals(ExpertCombination.COMBINATION_METHOD.ALL)) {
			for (ExpertCombination.COMBINATION_METHOD combinationMethod : ExpertCombination.COMBINATION_METHOD
					.values()) {
				if (combinationMethod.equals(ExpertCombination.COMBINATION_METHOD.ALL)) {
					continue;
				}
				int pass = passCounter.get(combinationMethod);
				int fail = failCounter.get(combinationMethod);
				double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

				System.out.println(combinationMethod + ";" + accuracy + ";" + pass + ";" + fail + ";;;;;;;;;;");
				outStringBuilder
				.append(combinationMethod + ";" + accuracy + ";" + pass + ";" + fail + ";;;;;;;;;;" + "\n");
			}
		} else {
			int pass = passCounter.get(combMethod);
			int fail = failCounter.get(combMethod);
			double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

			System.out.println(combMethod + ";" + accuracy + ";" + pass + ";" + fail);
			outStringBuilder.append(combMethod + ";" + accuracy + ";" + pass + ";" + fail);
		}

		double[] prec = new double[CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS];
		double[] f1_values = new double[CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS];
		double[] f1_values_original = new double[CIFAR10_DNNt_Combined.NUMBER_OF_EXPERTS];
		List<Integer> f1Experts = new ArrayList<>();
		StringBuilder bs = new StringBuilder();

		for (int expertId : expertIDs) {
			int pass = passCounter.get(expertId);
			int fail = failCounter.get(expertId);
			double accuracy = round((((double) pass) / (pass + fail)) * 100.0, 2);

			int targetedPass = targetedPassCounter.get(expertId);
			int targetedFail = targetedFailCounter.get(expertId);
			double targetedAccuracy = round((((double) targetedPass) / (targetedPass + targetedFail)) * 100.0, 2);

			int TP = TPCounter.get(expertId);
			int TN = TNCounter.get(expertId);
			int FP = FPCounter.get(expertId);
			int FN = FNCounter.get(expertId);
			double precision = ((double) TP) / (TP + FP);
			prec[expertId] = precision;

			double recall = ((double) TP) / (TP + FN);
			double f1 = 2 * precision * recall / (precision + recall);

			System.out.println("L" + expertId + ";" + accuracy + ";" + pass + ";" + fail + ";" + targetedAccuracy + ";"
					+ targetedPass + ";" + targetedFail + ";" + TP + ";" + TN + ";" + FP + ";" + FN + ";"
					+ round(precision * 100.0, 2) + ";" + round(recall * 100.0, 2) + ";" + round(f1 * 100.0, 2));
			outStringBuilder.append("L" + expertId + ";" + accuracy + ";" + pass + ";" + fail + ";" + targetedAccuracy
					+ ";" + targetedPass + ";" + targetedFail + ";" + TP + ";" + TN + ";" + FP + ";" + FN + ";"
					+ round(precision * 100.0, 2) + ";" + round(recall * 100.0, 2) + ";" + round(f1 * 100.0, 2) + "\n");

			String id = "ORIG_L" + expertId;

			int targetedPass_O = targetedPassCounter.get(id);
			int targetedFail_O = targetedFailCounter.get(id);
			double targetedAccuracy_O = round((((double) targetedPass_O) / (targetedPass_O + targetedFail_O)) * 100.0,
					2);

			int TP_O = TPCounter.get(id);
			int TN_O = TNCounter.get(id);
			int FP_O = FPCounter.get(id);
			int FN_O = FNCounter.get(id);

			double precision_O = ((double) TP_O) / (TP_O + FP_O);
			double recall_O = ((double) TP_O) / (TP_O + FN_O);
			double f1_O = 2 * precision_O * recall_O / (precision_O + recall_O);

			bs.append(id + ";;;;" + targetedAccuracy_O + ";" + targetedPass_O + ";" + targetedFail_O + ";" + TP_O + ";"
					+ TN_O + ";" + FP_O + ";" + FN_O + ";" + round(precision_O * 100.0, 2) + ";"
					+ round(recall_O * 100.0, 2) + ";" + round(f1_O * 100.0, 2) + "\n");

			if (f1 > f1_O) {
				f1Experts.add(expertId);
			}

			f1_values[expertId] = f1;
			f1_values_original[expertId] = f1_O;
		}

		System.out.println(bs.toString());
		outStringBuilder.append(bs.toString());

		System.out.println();
		outStringBuilder.append("\n");
		System.out.println("prec=" + Arrays.toString(prec));
		System.out.println("f1Experts=" + Arrays.toString(f1Experts.toArray()));
		System.out.println("f1_values=" + Arrays.toString(f1_values));
		System.out.println("f1_values_original=" + Arrays.toString(f1_values_original));

		if (subject.getOutputPath() != null) {
			BufferedWriter writer = new BufferedWriter(new FileWriter(subject.getOutputPath() + "/" + subject.toString()
			+ (useF1Selection ? "_f1" : "") + (useF1HarmonicSelection ? "_f1har" : "") + ".csv"));
			writer.write(outStringBuilder.toString());
			writer.close();

			writer = new BufferedWriter(new FileWriter(subject.getOutputPath() + "/" + subject.toString()
			+ (useF1Selection ? "_f1" : "") + (useF1HarmonicSelection ? "_f1har" : "") + "_prec_f1.csv"));
			writer.write("prec=" + Arrays.toString(prec) + "\n");
			writer.write("f1Experts=" + Arrays.toString(f1Experts.toArray()) + "\n");
			writer.write("f1_values=" + Arrays.toString(f1_values) + "\n");
			writer.write("f1_values_original=" + Arrays.toString(f1_values_original) + "\n");
			writer.close();
		}

	}

	public static void main(String[] args) {
		try {
			long startTime = System.currentTimeMillis();
			//			runMNIST0Experiment(SUBJECT.LOW_QUALITY_PATTERN_TRAINING, ExpertCombination.COMBINATION_METHOD.ALL);

			//			SUBJECT[] subjects = { SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_ADV_TRAINING, };

			//			SUBJECT[] subjects = { SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpB_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpB_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpB_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpC_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpC_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpC_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TRAINING, };
			//
			//			for (SUBJECT subject : subjects) {
			//				runMNIST0Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, false, false);
			//			}

			//			runMNIST0Experiment(SUBJECT.LOW_QUALITY_PATTERN_TEST, ExpertCombination.COMBINATION_METHOD.ALL, 60000,
			//					false, false);

			//			SUBJECT[] f1_subjects = { SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TRAINING, };
			//
			//			for (SUBJECT subject : f1_subjects) {
			//				runMNIST0Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, true, false);
			//			}
			//
			//			SUBJECT[] f1_harmonic_subjects = { SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpA_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpB_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpC_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_10_ExpD_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpD_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpA_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpB_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpC_TRAINING,
			//
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_ADV_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_ADV_TRAINING,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TEST,
			//					SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_05_ExpD_TRAINING, };
			//
			//			for (SUBJECT subject : f1_harmonic_subjects) {
			//				runMNIST0Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, false, true);
			//			}

			//			runMNIST0Experiment(SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_TEST,
			//					ExpertCombination.COMBINATION_METHOD.ALL, 60000, false);
			//			runMNIST0CombinationOverheadExperiment(SUBJECT.LOW_QUALITY_LAST_LAYER_TEST, 60000, false,
			//					new int[] { 6, 8, 9 });
			//			runOriginalCIFARDNN(SUBJECT.CIFAR_LAST_LAYER_ORIGINAL_TRAINING, 60000);
			//			runOriginalCIFARDNN(SUBJECT.CIFAR_LAST_LAYER_TRAINING, 60000);
			//			runCIFAR10Experiment(SUBJECT.CIFAR_LAST_LAYER_TEST, ExpertCombination.COMBINATION_METHOD.ALL, 10000, false);

			//			runMNIST0Experiment(SUBJECT.ADVERSARIAL_LAST_LAYER_Eps0_01_ExpA_TEST,
			//					ExpertCombination.COMBINATION_METHOD.ALL, 60000, false, false);

			//			SUBJECT[] subjects = {
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_ADV_TEST,
			//			};
			//			for (SUBJECT subject : subjects) {
			//				if (subject.toString().endsWith("_ADV_TRAINING")) { // For adversarial training only consider first 10000
			//					runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 10000, false, false);	
			//				} else {
			//					runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, false, false);
			//				}
			//			}

			//			SUBJECT[] f1_subjects = {
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_ADV_TEST,
			//			};
			//			for (SUBJECT subject : f1_subjects) {
			//				if (subject.toString().endsWith("_ADV_TRAINING")) { // For adversarial training only consider first 10000
			//					runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 10000, true, false);	
			//				} else {
			//					runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, true, false);
			//				}
			//			}
			//			
			//			SUBJECT[] f1_harmonic_subjects = {
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpA_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpB_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpC_ADV_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_TEST,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_ADV_TRAINING,
			//					SUBJECT.CIFAR_LAST_LAYER_Eps0_01_ExpD_ADV_TEST,
			//			};
			//			for (SUBJECT subject : f1_harmonic_subjects) {
			//				if (subject.toString().endsWith("_ADV_TRAINING")) { // For adversarial training only consider first 10000
			//					runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 10000, false, true);	
			//				} else {
			//					runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, false, true);
			//				}
			//			}

			// LOW_QUALITY_PATTERN_TEST
			// LOW_QUALITY_PATTERN_TRAINING

			// LOW_QUALITY_LAST_LAYER_TEST
			// LOW_QUALITY_LAST_LAYER_TRAINING

			//			int numberOfInputs = 10000; // 10000
			//			int iterations = 1000;
			//			runMNIST0CombinationOverheadExperiment(SUBJECT.LOW_QUALITY_LAST_LAYER_TEST, numberOfInputs, iterations, false);
			//			runMNIST0CombinationOverheadExperiment(SUBJECT.LOW_QUALITY_PATTERN_TEST, numberOfInputs, iterations, false);
			//			runMNIST0CombinationOverheadExperiment(SUBJECT.LOW_QUALITY_PATTERN_TEST, numberOfInputs, iterations, true);

			////////////////////////////////////////////////////////////////////////////////////////////////////

			SUBJECT[] subjects = { 
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TRAINING,

					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_POISONED_TRAINING,


					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_POISONED_TRAINING,


					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_POISONED_TRAINING,

			};

			for (SUBJECT subject : subjects) {
	//			runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, false, false);
			}

			SUBJECT[] f1_subjects = {
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TRAINING,

					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_POISONED_TRAINING,


		//			SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
		//			SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_POISONED_TEST,
		//			SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_POISONED_TRAINING,


					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_POISONED_TRAINING,
		};
			for (SUBJECT subject : f1_subjects) {
							runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, true, false);
			}

			SUBJECT[] f1_harmonic_subjects = { 
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_POISONED_TRAINING,

					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpB_POISONED_TRAINING,


					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpC_POISONED_TRAINING,


					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_TEST,
					//SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpA_TRAINING,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_POISONED_TEST,
					SUBJECT.POISONED_CIFAR_LAST_LAYER_ExpD_POISONED_TRAINING,
		};
			for (SUBJECT subject : f1_harmonic_subjects) {
				//		runCIFAR10Experiment(subject, ExpertCombination.COMBINATION_METHOD.ALL, 60000, false, true);
			}

			long totalRuntime = System.currentTimeMillis() - startTime;
			System.out.println();
			System.out.println("Total Runtime: " + totalRuntime + " ms");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * DNNt program that takes repaired weights as input (currently as z3 output).
 */
public class CIFAR10_DNNt_Combined {

	/*
	 * *****************************************************************************
	 * Repaired Network Implementation
	 * *****************************************************************************
	 */

	public final static int NUMBER_OF_EXPERTS = 10;

	private CIFAR10_InternalData internal;
	private Object weight_delta;

	public CIFAR10_DNNt_Combined(CIFAR10_InternalData internal, Object repaired_weight_deltas) throws IOException {
		this.internal = internal;
		this.weight_delta = repaired_weight_deltas;
	}

	Map<Integer, double[][][]> layer_0(double[][][] input, int repairedLayerId, int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 0;
		Map<Integer, double[][][]> layer0_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer0_orig = new double[30][30][32];
		for (int i = 0; i < 30; i++)
			for (int j = 0; j < 30; j++)
				for (int k = 0; k < 32; k++) {
					layer0_orig[i][j][k] = internal.biases0[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 3; K++)
								layer0_orig[i][j][k] += internal.weights0[I][J][K][k] * input[i + I][j + J][K];
				}
		layer0_perExpert.put(-1, layer0_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][][][] delta_layer0_perExpert = (double[][][][][]) weight_delta;

			for (int expertId : expertIDs) {
				double[][][] layer0 = new double[30][30][32];
				for (int i = 0; i < 30; i++)
					for (int j = 0; j < 30; j++)
						for (int k = 0; k < 32; k++) {
							layer0_orig[i][j][k] = internal.biases0[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 3; K++)
										layer0[i][j][k] += (internal.weights0[I][J][K][k]
												+ delta_layer0_perExpert[expertId][I][J][K][k])
												* input[i + I][j + J][K];
						}
				layer0_perExpert.put(expertId, layer0);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer0 = new double[30][30][32];
					for (int i = 0; i < 30; i++)
						for (int j = 0; j < 30; j++)
							for (int k = 0; k < 32; k++) {
								layer0_orig[i][j][k] = internal.biases0[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 3; K++)
											layer0[i][j][k] += (internal.weights0[I][J][K][k]
													+ delta_layer0_perExpert[expertId][I][J][K][k])
													* input[i + I][j + J][K];
							}
					layer0_perExpert.put(expertId, layer0);
				}
			}
		}

		return layer0_perExpert;
	}

	Map<Integer, double[][][]> layer_1(Map<Integer, double[][][]> layer0_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 1;
		Map<Integer, double[][][]> layer1_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer0_orig = layer0_perExpert.get(-1);
		double[][][] layer1_orig = new double[30][30][32];
		for (int i = 0; i < 30; i++)
			for (int j = 0; j < 30; j++)
				for (int k = 0; k < 32; k++)
					if (layer0_orig[i][j][k] > 0)
						layer1_orig[i][j][k] = layer0_orig[i][j][k];
					else
						layer1_orig[i][j][k] = 0;
		layer1_perExpert.put(-1, layer1_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer0 = layer0_perExpert.get(expertId);

				double[][][] layer1 = new double[30][30][32];
				for (int i = 0; i < 30; i++)
					for (int j = 0; j < 30; j++)
						for (int k = 0; k < 32; k++)
							if (layer0[i][j][k] > 0)
								layer1[i][j][k] = layer0[i][j][k];
							else
								layer1[i][j][k] = 0;

				layer1_perExpert.put(expertId, layer1);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer0 = layer0_perExpert.get(expertId);

					double[][][] layer1 = new double[30][30][32];
					for (int i = 0; i < 30; i++)
						for (int j = 0; j < 30; j++)
							for (int k = 0; k < 32; k++)
								if (layer0[i][j][k] > 0)
									layer1[i][j][k] = layer0[i][j][k];
								else
									layer1[i][j][k] = 0;

					layer1_perExpert.put(expertId, layer1);
				}
			}
		}

		return layer1_perExpert;
	}

	Map<Integer, double[][][]> layer_2(Map<Integer, double[][][]> layer1_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 2;
		Map<Integer, double[][][]> layer2_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer1_orig = layer1_perExpert.get(-1);
		double[][][] layer2_orig = new double[28][28][32];
		for (int i = 0; i < 28; i++)
			for (int j = 0; j < 28; j++)
				for (int k = 0; k < 32; k++) {
					layer2_orig[i][j][k] = internal.biases2[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 32; K++)
								layer2_orig[i][j][k] += internal.weights2[I][J][K][k] * layer1_orig[i + I][j + J][K];
				}
		layer2_perExpert.put(-1, layer2_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][][][] delta_layer2_perExpert = (double[][][][][]) weight_delta;

			for (int expertId : expertIDs) {
				double[][][] layer2 = new double[28][28][32];
				for (int i = 0; i < 28; i++)
					for (int j = 0; j < 28; j++)
						for (int k = 0; k < 32; k++) {
							layer2[i][j][k] = internal.biases2[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 32; K++)
										layer2[i][j][k] += (internal.weights2[I][J][K][k]
												+ delta_layer2_perExpert[expertId][I][J][K][k])
												* layer1_orig[i + I][j + J][K];
						}

				layer2_perExpert.put(expertId, layer2);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer2 = new double[28][28][32];
					for (int i = 0; i < 28; i++)
						for (int j = 0; j < 28; j++)
							for (int k = 0; k < 32; k++) {
								layer2[i][j][k] = internal.biases2[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 32; K++)
											layer2[i][j][k] += (internal.weights2[I][J][K][k]
													+ delta_layer2_perExpert[expertId][I][J][K][k])
													* layer1_orig[i + I][j + J][K];
							}

					layer2_perExpert.put(expertId, layer2);
				}
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer1 = layer1_perExpert.get(expertId);

				double[][][] layer2 = new double[28][28][32];
				for (int i = 0; i < 28; i++)
					for (int j = 0; j < 28; j++)
						for (int k = 0; k < 32; k++) {
							layer2[i][j][k] = internal.biases2[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 32; K++)
										layer2[i][j][k] += internal.weights2[I][J][K][k] * layer1[i + I][j + J][K];
						}

				layer2_perExpert.put(expertId, layer2);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer1 = layer1_perExpert.get(expertId);

					double[][][] layer2 = new double[28][28][32];
					for (int i = 0; i < 28; i++)
						for (int j = 0; j < 28; j++)
							for (int k = 0; k < 32; k++) {
								layer2[i][j][k] = internal.biases2[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 32; K++)
											layer2[i][j][k] += internal.weights2[I][J][K][k] * layer1[i + I][j + J][K];
							}

					layer2_perExpert.put(expertId, layer2);
				}
			}
		}

		return layer2_perExpert;
	}

	Map<Integer, double[][][]> layer_3(Map<Integer, double[][][]> layer2_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 3;
		Map<Integer, double[][][]> layer3_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer2_orig = layer2_perExpert.get(-1);
		double[][][] layer3_orig = new double[28][28][32];
		for (int i = 0; i < 28; i++)
			for (int j = 0; j < 28; j++)
				for (int k = 0; k < 32; k++)
					if (layer2_orig[i][j][k] > 0)
						layer3_orig[i][j][k] = layer2_orig[i][j][k];
					else
						layer3_orig[i][j][k] = 0;
		layer3_perExpert.put(-1, layer3_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer2 = layer2_perExpert.get(expertId);

				double[][][] layer3 = new double[28][28][32];
				for (int i = 0; i < 28; i++)
					for (int j = 0; j < 28; j++)
						for (int k = 0; k < 32; k++)
							if (layer2[i][j][k] > 0)
								layer3[i][j][k] = layer2[i][j][k];
							else
								layer3[i][j][k] = 0;

				layer3_perExpert.put(expertId, layer3);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer2 = layer2_perExpert.get(expertId);

					double[][][] layer3 = new double[28][28][32];
					for (int i = 0; i < 28; i++)
						for (int j = 0; j < 28; j++)
							for (int k = 0; k < 32; k++)
								if (layer2[i][j][k] > 0)
									layer3[i][j][k] = layer2[i][j][k];
								else
									layer3[i][j][k] = 0;

					layer3_perExpert.put(expertId, layer3);
				}
			}
		}

		return layer3_perExpert;
	}

	Map<Integer, double[][][]> layer_4(Map<Integer, double[][][]> layer3_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 4;
		Map<Integer, double[][][]> layer4_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer3_orig = layer3_perExpert.get(-1);
		double[][][] layer4_orig = new double[14][14][32];
		for (int i = 0; i < 14; i++)
			for (int j = 0; j < 14; j++)
				for (int k = 0; k < 32; k++) {
					layer4_orig[i][j][k] = 0;
					for (int I = i * 2; I < (i + 1) * 2; I++)
						for (int J = j * 2; J < (j + 1) * 2; J++)
							if (layer3_orig[I][J][k] > layer4_orig[i][j][k])
								layer4_orig[i][j][k] = layer3_orig[I][J][k];
				}
		layer4_perExpert.put(-1, layer4_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer3 = layer3_perExpert.get(expertId);

				double[][][] layer4 = new double[14][14][32];
				for (int i = 0; i < 14; i++)
					for (int j = 0; j < 14; j++)
						for (int k = 0; k < 32; k++) {
							layer4[i][j][k] = 0;
							for (int I = i * 2; I < (i + 1) * 2; I++)
								for (int J = j * 2; J < (j + 1) * 2; J++)
									if (layer3[I][J][k] > layer4_orig[i][j][k])
										layer4[i][j][k] = layer3[I][J][k];
						}

				layer4_perExpert.put(expertId, layer4);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer3 = layer3_perExpert.get(expertId);

					double[][][] layer4 = new double[14][14][32];
					for (int i = 0; i < 14; i++)
						for (int j = 0; j < 14; j++)
							for (int k = 0; k < 32; k++) {
								layer4[i][j][k] = 0;
								for (int I = i * 2; I < (i + 1) * 2; I++)
									for (int J = j * 2; J < (j + 1) * 2; J++)
										if (layer3[I][J][k] > layer4_orig[i][j][k])
											layer4[i][j][k] = layer3[I][J][k];
							}

					layer4_perExpert.put(expertId, layer4);
				}
			}
		}

		return layer4_perExpert;
	}

	Map<Integer, double[][][]> layer_5(Map<Integer, double[][][]> layer4_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 5;
		Map<Integer, double[][][]> layer5_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer4_orig = layer4_perExpert.get(-1);
		double[][][] layer5_orig = new double[12][12][64];
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				for (int k = 0; k < 64; k++) {
					layer5_orig[i][j][k] = internal.biases5[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 32; K++)
								layer5_orig[i][j][k] += internal.weights5[I][J][K][k] * layer4_orig[i + I][j + J][K];
				}
		layer5_perExpert.put(-1, layer5_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][][][] delta_layer5_perExpert = (double[][][][][]) weight_delta;

			for (int expertId : expertIDs) {

				double[][][] layer5 = new double[12][12][64];
				for (int i = 0; i < 12; i++)
					for (int j = 0; j < 12; j++)
						for (int k = 0; k < 64; k++) {
							layer5[i][j][k] = internal.biases5[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 32; K++)
										layer5[i][j][k] += (internal.weights5[I][J][K][k]
												+ delta_layer5_perExpert[expertId][I][J][K][k])
												* layer4_orig[i + I][j + J][K];
						}

				layer5_perExpert.put(expertId, layer5);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer5 = new double[12][12][64];
					for (int i = 0; i < 12; i++)
						for (int j = 0; j < 12; j++)
							for (int k = 0; k < 64; k++) {
								layer5[i][j][k] = internal.biases5[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 32; K++)
											layer5[i][j][k] += (internal.weights5[I][J][K][k]
													+ delta_layer5_perExpert[expertId][I][J][K][k])
													* layer4_orig[i + I][j + J][K];
							}

					layer5_perExpert.put(expertId, layer5);
				}
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer4 = layer4_perExpert.get(expertId);

				double[][][] layer5 = new double[12][12][64];
				for (int i = 0; i < 12; i++)
					for (int j = 0; j < 12; j++)
						for (int k = 0; k < 64; k++) {
							layer5[i][j][k] = internal.biases5[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 32; K++)
										layer5[i][j][k] += internal.weights5[I][J][K][k] * layer4[i + I][j + J][K];
						}

				layer5_perExpert.put(expertId, layer5);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer4 = layer4_perExpert.get(expertId);

					double[][][] layer5 = new double[12][12][64];
					for (int i = 0; i < 12; i++)
						for (int j = 0; j < 12; j++)
							for (int k = 0; k < 64; k++) {
								layer5[i][j][k] = internal.biases5[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 32; K++)
											layer5[i][j][k] += internal.weights5[I][J][K][k] * layer4[i + I][j + J][K];
							}

					layer5_perExpert.put(expertId, layer5);
				}
			}
		}

		return layer5_perExpert;
	}

	Map<Integer, double[][][]> layer_6(Map<Integer, double[][][]> layer5_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 6;
		Map<Integer, double[][][]> layer6_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer5_orig = layer5_perExpert.get(-1);
		double[][][] layer6_orig = new double[12][12][64];
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				for (int k = 0; k < 64; k++)
					if (layer5_orig[i][j][k] > 0)
						layer6_orig[i][j][k] = layer5_orig[i][j][k];
					else
						layer6_orig[i][j][k] = 0;
		layer6_perExpert.put(-1, layer6_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer5 = layer5_perExpert.get(expertId);

				double[][][] layer6 = new double[12][12][64];
				for (int i = 0; i < 12; i++)
					for (int j = 0; j < 12; j++)
						for (int k = 0; k < 64; k++)
							if (layer5[i][j][k] > 0)
								layer6[i][j][k] = layer5[i][j][k];
							else
								layer6[i][j][k] = 0;

				layer6_perExpert.put(expertId, layer6);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer5 = layer5_perExpert.get(expertId);

					double[][][] layer6 = new double[12][12][64];
					for (int i = 0; i < 12; i++)
						for (int j = 0; j < 12; j++)
							for (int k = 0; k < 64; k++)
								if (layer5[i][j][k] > 0)
									layer6[i][j][k] = layer5[i][j][k];
								else
									layer6[i][j][k] = 0;

					layer6_perExpert.put(expertId, layer6);
				}
			}
		}

		return layer6_perExpert;
	}

	Map<Integer, double[][][]> layer_7(Map<Integer, double[][][]> layer6_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 7;
		Map<Integer, double[][][]> layer7_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer6_orig = layer6_perExpert.get(-1);
		double[][][] layer7_orig = new double[10][10][64];
		for (int i = 0; i < 10; i++)
			for (int j = 0; j < 10; j++)
				for (int k = 0; k < 64; k++) {
					layer7_orig[i][j][k] = internal.biases7[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 64; K++)
								layer7_orig[i][j][k] += internal.weights7[I][J][K][k] * layer6_orig[i + I][j + J][K];
				}

		layer7_perExpert.put(-1, layer7_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][][][] delta_layer7_perExpert = (double[][][][][]) weight_delta;

			for (int expertId : expertIDs) {
				double[][][] layer7 = new double[10][10][64];
				for (int i = 0; i < 10; i++)
					for (int j = 0; j < 10; j++)
						for (int k = 0; k < 64; k++) {
							layer7[i][j][k] = internal.biases7[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 64; K++)
										layer7[i][j][k] += (internal.weights7[I][J][K][k]
												+ delta_layer7_perExpert[expertId][I][J][K][k])
												* layer6_orig[i + I][j + J][K];
						}

				layer7_perExpert.put(expertId, layer7);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer7 = new double[10][10][64];
					for (int i = 0; i < 10; i++)
						for (int j = 0; j < 10; j++)
							for (int k = 0; k < 64; k++) {
								layer7[i][j][k] = internal.biases7[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 64; K++)
											layer7[i][j][k] += (internal.weights7[I][J][K][k]
													+ delta_layer7_perExpert[expertId][I][J][K][k])
													* layer6_orig[i + I][j + J][K];
							}

					layer7_perExpert.put(expertId, layer7);
				}
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer6 = layer6_perExpert.get(expertId);

				double[][][] layer7 = new double[10][10][64];
				for (int i = 0; i < 10; i++)
					for (int j = 0; j < 10; j++)
						for (int k = 0; k < 64; k++) {
							layer7[i][j][k] = internal.biases7[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 64; K++)
										layer7[i][j][k] += internal.weights7[I][J][K][k] * layer6[i + I][j + J][K];
						}

				layer7_perExpert.put(expertId, layer7);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer6 = layer6_perExpert.get(expertId);

					double[][][] layer7 = new double[10][10][64];
					for (int i = 0; i < 10; i++)
						for (int j = 0; j < 10; j++)
							for (int k = 0; k < 64; k++) {
								layer7[i][j][k] = internal.biases7[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 64; K++)
											layer7[i][j][k] += internal.weights7[I][J][K][k] * layer6[i + I][j + J][K];
							}

					layer7_perExpert.put(expertId, layer7);
				}
			}
		}

		return layer7_perExpert;
	}

	Map<Integer, double[][][]> layer_8(Map<Integer, double[][][]> layer7_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 8;
		Map<Integer, double[][][]> layer8_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer7_orig = layer7_perExpert.get(-1);
		double[][][] layer8_orig = new double[10][10][64];
		for (int i = 0; i < 10; i++)
			for (int j = 0; j < 10; j++)
				for (int k = 0; k < 64; k++)
					if (layer7_orig[i][j][k] > 0)
						layer8_orig[i][j][k] = layer7_orig[i][j][k];
					else
						layer8_orig[i][j][k] = 0;
		layer8_perExpert.put(-1, layer8_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer7 = layer7_perExpert.get(expertId);

				double[][][] layer8 = new double[10][10][64];
				for (int i = 0; i < 10; i++)
					for (int j = 0; j < 10; j++)
						for (int k = 0; k < 64; k++)
							if (layer7[i][j][k] > 0)
								layer8[i][j][k] = layer7[i][j][k];
							else
								layer8[i][j][k] = 0;

				layer8_perExpert.put(expertId, layer8);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer7 = layer7_perExpert.get(expertId);

					double[][][] layer8 = new double[10][10][64];
					for (int i = 0; i < 10; i++)
						for (int j = 0; j < 10; j++)
							for (int k = 0; k < 64; k++)
								if (layer7[i][j][k] > 0)
									layer8[i][j][k] = layer7[i][j][k];
								else
									layer8[i][j][k] = 0;

					layer8_perExpert.put(expertId, layer8);
				}
			}
		}

		return layer8_perExpert;
	}

	Map<Integer, double[][][]> layer_9(Map<Integer, double[][][]> layer8_perExpert, int repairedLayerId,
			int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 9;
		Map<Integer, double[][][]> layer9_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer8_orig = layer8_perExpert.get(-1);
		double[][][] layer9_orig = new double[5][5][64];
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				for (int k = 0; k < 64; k++) {
					layer9_orig[i][j][k] = 0;
					for (int I = i * 2; I < (i + 1) * 2; I++)
						for (int J = j * 2; J < (j + 1) * 2; J++)
							if (layer8_orig[I][J][k] > layer9_orig[i][j][k])
								layer9_orig[i][j][k] = layer8_orig[I][J][k];
				}
		layer9_perExpert.put(-1, layer9_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer8 = layer8_perExpert.get(expertId);

				double[][][] layer9 = new double[5][5][64];
				for (int i = 0; i < 5; i++)
					for (int j = 0; j < 5; j++)
						for (int k = 0; k < 64; k++) {
							layer9[i][j][k] = 0;
							for (int I = i * 2; I < (i + 1) * 2; I++)
								for (int J = j * 2; J < (j + 1) * 2; J++)
									if (layer8[I][J][k] > layer9[i][j][k])
										layer9[i][j][k] = layer8[I][J][k];
						}

				layer9_perExpert.put(expertId, layer9);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer8 = layer8_perExpert.get(expertId);

					double[][][] layer9 = new double[5][5][64];
					for (int i = 0; i < 5; i++)
						for (int j = 0; j < 5; j++)
							for (int k = 0; k < 64; k++) {
								layer9[i][j][k] = 0;
								for (int I = i * 2; I < (i + 1) * 2; I++)
									for (int J = j * 2; J < (j + 1) * 2; J++)
										if (layer8[I][J][k] > layer9[i][j][k])
											layer9[i][j][k] = layer8[I][J][k];
							}

					layer9_perExpert.put(expertId, layer9);
				}
			}
		}

		return layer9_perExpert;
	}

	Map<Integer, double[]> layer_10(Map<Integer, double[][][]> layer9_perExpert, int repairedLayerId, int[] expertIDs,
			boolean optimized) {
		final int CURRENT_LAYER = 10;
		Map<Integer, double[]> layer10_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer9_orig = layer9_perExpert.get(-1);
		double[] layer10_orig = new double[1600];
		for (int i = 0; i < 1600; i++) {
			int d0 = i / 320;
			int d1 = (i % 320) / 64;
			int d2 = i - d0 * 320 - d1 * 64;
			layer10_orig[i] = layer9_orig[d0][d1][d2];
		}
		layer10_perExpert.put(-1, layer10_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer9 = layer9_perExpert.get(expertId);

				double[] layer10 = new double[1600];
				for (int i = 0; i < 1600; i++) {
					int d0 = i / 320;
					int d1 = (i % 320) / 64;
					int d2 = i - d0 * 320 - d1 * 64;
					layer10[i] = layer9[d0][d1][d2];
				}

				layer10_perExpert.put(expertId, layer10);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer9 = layer9_perExpert.get(expertId);

					double[] layer10 = new double[1600];
					for (int i = 0; i < 1600; i++) {
						int d0 = i / 320;
						int d1 = (i % 320) / 64;
						int d2 = i - d0 * 320 - d1 * 64;
						layer10[i] = layer9[d0][d1][d2];
					}

					layer10_perExpert.put(expertId, layer10);
				}
			}
		}

		return layer10_perExpert;
	}

	Map<Integer, double[]> layer_11(Map<Integer, double[]> layer10_perExpert, int repairedLayerId, int[] expertIDs,
			boolean optimized) {
		final int CURRENT_LAYER = 11;
		Map<Integer, double[]> layer11_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer10_orig = layer10_perExpert.get(-1);
		double[] layer11_orig = new double[512];
		for (int i = 0; i < 512; i++) {
			layer11_orig[i] = internal.biases11[i];
			for (int I = 0; I < 1600; I++)
				layer11_orig[i] += internal.weights11[I][i] * layer10_orig[I];
		}
		layer11_perExpert.put(-1, layer11_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][] delta_layer11_perExpert = (double[][][]) weight_delta;

			for (int expertId : expertIDs) {
				double[] layer11 = new double[512];
				for (int i = 0; i < 512; i++) {
					layer11[i] = internal.biases11[i];
					for (int I = 0; I < 1600; I++)
						layer11[i] += (internal.weights11[I][i] + delta_layer11_perExpert[expertId][I][i])
								* layer10_orig[I];
				}

				layer11_perExpert.put(expertId, layer11);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer11 = new double[512];
					for (int i = 0; i < 512; i++) {
						layer11[i] = internal.biases11[i];
						for (int I = 0; I < 1600; I++)
							layer11[i] += (internal.weights11[I][i] + delta_layer11_perExpert[expertId][I][i])
									* layer10_orig[I];
					}

					layer11_perExpert.put(expertId, layer11);
				}
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId : expertIDs) {
				double[] layer10 = layer10_perExpert.get(expertId);

				double[] layer11 = new double[512];
				for (int i = 0; i < 512; i++) {
					layer11[i] = internal.biases11[i];
					for (int I = 0; I < 1600; I++)
						layer11[i] += internal.weights11[I][i] * layer10[I];
				}

				layer11_perExpert.put(expertId, layer11);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer10 = layer10_perExpert.get(expertId);

					double[] layer11 = new double[512];
					for (int i = 0; i < 512; i++) {
						layer11[i] = internal.biases11[i];
						for (int I = 0; I < 1600; I++)
							layer11[i] += internal.weights11[I][i] * layer10[I];
					}

					layer11_perExpert.put(expertId, layer11);
				}
			}
		}

		return layer11_perExpert;
	}

	Map<Integer, double[]> layer_12(Map<Integer, double[]> layer11_perExpert, int repairedLayerId, int[] expertIDs,
			boolean optimized) {
		final int CURRENT_LAYER = 12;
		Map<Integer, double[]> layer12_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer11_orig = layer11_perExpert.get(-1);
		double[] layer12_orig = new double[512];
		for (int i = 0; i < 512; i++)
			if (layer11_orig[i] > 0)
				layer12_orig[i] = layer11_orig[i];
			else
				layer12_orig[i] = 0;
		layer12_perExpert.put(-1, layer12_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[] layer11 = layer11_perExpert.get(expertId);

				double[] layer12 = new double[512];
				for (int i = 0; i < 512; i++)
					if (layer11[i] > 0)
						layer12[i] = layer11[i];
					else
						layer12[i] = 0;

				layer12_perExpert.put(expertId, layer12);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer11 = layer11_perExpert.get(expertId);

					double[] layer12 = new double[512];
					for (int i = 0; i < 512; i++)
						if (layer11[i] > 0)
							layer12[i] = layer11[i];
						else
							layer12[i] = 0;

					layer12_perExpert.put(expertId, layer12);
				}
			}
		}

		return layer12_perExpert;
	}

	Map<Integer, double[]> layer_13(Map<Integer, double[]> layer12_perExpert, int repairedLayerId, int[] expertIDs,
			boolean optimized) {
		final int CURRENT_LAYER = 13;
		Map<Integer, double[]> layer13_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer12_orig = layer12_perExpert.get(-1);
		double[] layer13_orig = new double[10];
		for (int i = 0; i < 10; i++) {
			layer13_orig[i] = internal.biases13[i];
			for (int I = 0; I < 512; I++)
				layer13_orig[i] += internal.weights13[I][i] * layer12_orig[I];
		}
		layer13_perExpert.put(-1, layer13_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][] delta_layer13_perExpert = (double[][][]) weight_delta;

			for (int expertId : expertIDs) {
				double[] layer13 = new double[10];
				for (int i = 0; i < 10; i++) {
					layer13[i] = internal.biases13[i];
					for (int I = 0; I < 512; I++)
						layer13[i] += (internal.weights13[I][i] + delta_layer13_perExpert[expertId][I][i])
								* layer12_orig[I];
				}

				layer13_perExpert.put(expertId, layer13);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer13 = new double[10];
					for (int i = 0; i < 10; i++) {
						layer13[i] = internal.biases13[i];
						for (int I = 0; I < 512; I++)
							layer13[i] += (internal.weights13[I][i] + delta_layer13_perExpert[expertId][I][i])
									* layer12_orig[I];
					}

					layer13_perExpert.put(expertId, layer13);
				}
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId : expertIDs) {
				double[] layer12 = layer12_perExpert.get(expertId);

				double[] layer13 = new double[10];
				for (int i = 0; i < 10; i++) {
					layer13[i] = internal.biases13[i];
					for (int I = 0; I < 512; I++)
						layer13[i] += internal.weights13[I][i] * layer12[I];
				}

				layer13_perExpert.put(expertId, layer13);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer12 = layer12_perExpert.get(expertId);

					double[] layer13 = new double[10];
					for (int i = 0; i < 10; i++) {
						layer13[i] = internal.biases13[i];
						for (int I = 0; I < 512; I++)
							layer13[i] += internal.weights13[I][i] * layer12[I];
					}

					layer13_perExpert.put(expertId, layer13);
				}
			}
		}

		return layer13_perExpert;
	}

	/**
	 * Executes the repaired network with the given input. The executions assumes
	 * that the parameter repairedLayerId specifies the repaired layer.
	 * 
	 * @param input
	 * @param repairedLayerId
	 * @param expertIDs
	 * @return Mapping from expert network to values at the last layer.
	 * @throws IOException
	 */
	Map<Integer, double[]> run(double[][][] input, int repairedLayerId, int[] expertIDs) throws IOException {
		return run(input, repairedLayerId, expertIDs, false);
	}

	Map<Integer, double[]> run(double[][][] input, int repairedLayerId, int[] expertIDs, boolean optimized)
			throws IOException {

		// layer 0: conv2d_1
		Map<Integer, double[][][]> layer0_perExpert = layer_0(input, repairedLayerId, expertIDs, optimized);

		// layer 1: activation_1
		Map<Integer, double[][][]> layer1_perExpert = layer_1(layer0_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 2: conv2d_2
		Map<Integer, double[][][]> layer2_perExpert = layer_2(layer1_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 3: activation_2
		Map<Integer, double[][][]> layer3_perExpert = layer_3(layer2_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 4: max_pooling2d_1
		Map<Integer, double[][][]> layer4_perExpert = layer_4(layer3_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 5: conv2d_3
		Map<Integer, double[][][]> layer5_perExpert = layer_5(layer4_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 6: activation_3
		Map<Integer, double[][][]> layer6_perExpert = layer_6(layer5_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 7: conv2d_4
		Map<Integer, double[][][]> layer7_perExpert = layer_7(layer6_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 8: activation_4
		Map<Integer, double[][][]> layer8_perExpert = layer_8(layer7_perExpert, repairedLayerId, expertIDs, optimized);
		
		// layer 9: max_pooling2d_2
		Map<Integer, double[][][]> layer9_perExpert = layer_9(layer8_perExpert, repairedLayerId, expertIDs, optimized);
		
		// layer 10: flatten_1
		Map<Integer, double[]> layer10_perExpert = layer_10(layer9_perExpert, repairedLayerId, expertIDs, optimized);
		
		// layer 11: dense_1
		Map<Integer, double[]> layer11_perExpert = layer_11(layer10_perExpert, repairedLayerId, expertIDs, optimized);
		
		// layer 12: activation_5
		Map<Integer, double[]> layer12_perExpert = layer_12(layer11_perExpert, repairedLayerId, expertIDs, optimized);
		
		// layer 13: dense_2
		Map<Integer, double[]> layer13_perExpert = layer_13(layer12_perExpert, repairedLayerId, expertIDs, optimized);

		/*
		 * At this point we have layer13_perExpert with the original calculation results
		 * on position -1, and if there been a repaired layer, then the expert results
		 * are stored in positions 0-9.
		 */

		return layer13_perExpert;
	}

}

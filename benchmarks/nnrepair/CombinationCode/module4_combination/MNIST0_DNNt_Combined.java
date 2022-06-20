import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * DNNt program that takes repaired weights as input (currently as z3 output).
 */
public class MNIST0_DNNt_Combined {

	/*
	 * *****************************************************************************
	 * Repaired Network Implementation
	 * *****************************************************************************
	 */

	public final static int NUMBER_OF_EXPERTS = 10;

	private MNIST0_InternalData internal;
	private Object weight_delta;

	public MNIST0_DNNt_Combined(MNIST0_InternalData internal, Object repaired_weight_deltas) throws IOException {
		this.internal = internal;
		this.weight_delta = repaired_weight_deltas;
	}

	Map<Integer, double[][][]> layer_0(double[][][] input, int repairedLayerId, int[] expertIDs, boolean optimized) {
		final int CURRENT_LAYER = 0;
		Map<Integer, double[][][]> layer0_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer0_orig = new double[26][26][2];
		for (int i = 0; i < 26; i++)
			for (int j = 0; j < 26; j++)
				for (int k = 0; k < 2; k++) {
					layer0_orig[i][j][k] = internal.biases0[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 1; K++)
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
				double[][][] layer0 = new double[26][26][2];
				for (int i = 0; i < 26; i++)
					for (int j = 0; j < 26; j++)
						for (int k = 0; k < 2; k++) {
							layer0[i][j][k] = internal.biases0[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 1; K++)
										layer0[i][j][k] += (internal.weights0[I][J][K][k]
												+ delta_layer0_perExpert[expertId][I][J][K][k]) * input[i + I][j + J][K];
						}

				layer0_perExpert.put(expertId, layer0);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer0 = new double[26][26][2];
					for (int i = 0; i < 26; i++)
						for (int j = 0; j < 26; j++)
							for (int k = 0; k < 2; k++) {
								layer0[i][j][k] = internal.biases0[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 1; K++)
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
		double[][][] layer1_orig = new double[26][26][2];
		for (int i = 0; i < 26; i++)
			for (int j = 0; j < 26; j++)
				for (int k = 0; k < 2; k++)
					if (layer0_orig[i][j][k] > 0) {
						layer1_orig[i][j][k] = layer0_orig[i][j][k];

					} else {
						layer1_orig[i][j][k] = 0;

					}
		layer1_perExpert.put(-1, layer1_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer0 = layer0_perExpert.get(expertId);

				double[][][] layer1 = new double[26][26][2];
				for (int i = 0; i < 26; i++)
					for (int j = 0; j < 26; j++)
						for (int k = 0; k < 2; k++)
							if (layer0[i][j][k] > 0) {
								layer1[i][j][k] = layer0[i][j][k];

							} else {
								layer1[i][j][k] = 0;

							}

				layer1_perExpert.put(expertId, layer1);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer0 = layer0_perExpert.get(expertId);

					double[][][] layer1 = new double[26][26][2];
					for (int i = 0; i < 26; i++)
						for (int j = 0; j < 26; j++)
							for (int k = 0; k < 2; k++)
								if (layer0[i][j][k] > 0) {
									layer1[i][j][k] = layer0[i][j][k];

								} else {
									layer1[i][j][k] = 0;

								}

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
		double[][][] layer2_orig = new double[24][24][4];
		for (int i = 0; i < 24; i++)
			for (int j = 0; j < 24; j++)
				for (int k = 0; k < 4; k++) {
					layer2_orig[i][j][k] = internal.biases2[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 2; K++)
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
				double[][][] layer2 = new double[24][24][4];
				for (int i = 0; i < 24; i++)
					for (int j = 0; j < 24; j++)
						for (int k = 0; k < 4; k++) {
							layer2[i][j][k] = internal.biases2[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 2; K++)
										layer2[i][j][k] += (internal.weights2[I][J][K][k]
												+ delta_layer2_perExpert[expertId][I][J][K][k])
												* layer1_orig[i + I][j + J][K];
						}

				layer2_perExpert.put(expertId, layer2);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer2 = new double[24][24][4];
					for (int i = 0; i < 24; i++)
						for (int j = 0; j < 24; j++)
							for (int k = 0; k < 4; k++) {
								layer2[i][j][k] = internal.biases2[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 2; K++)
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

				double[][][] layer2 = new double[24][24][4];
				for (int i = 0; i < 24; i++)
					for (int j = 0; j < 24; j++)
						for (int k = 0; k < 4; k++) {
							layer2[i][j][k] = internal.biases2[k];
							for (int I = 0; I < 3; I++)
								for (int J = 0; J < 3; J++)
									for (int K = 0; K < 2; K++)
										layer2[i][j][k] += internal.weights2[I][J][K][k] * layer1[i + I][j + J][K];
						}

				layer2_perExpert.put(expertId, layer2);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer1 = layer1_perExpert.get(expertId);

					double[][][] layer2 = new double[24][24][4];
					for (int i = 0; i < 24; i++)
						for (int j = 0; j < 24; j++)
							for (int k = 0; k < 4; k++) {
								layer2[i][j][k] = internal.biases2[k];
								for (int I = 0; I < 3; I++)
									for (int J = 0; J < 3; J++)
										for (int K = 0; K < 2; K++)
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
		double[][][] layer3_orig = new double[24][24][4];
		for (int i = 0; i < 24; i++)
			for (int j = 0; j < 24; j++)
				for (int k = 0; k < 4; k++)
					if (layer2_orig[i][j][k] > 0) {
						layer3_orig[i][j][k] = layer2_orig[i][j][k];

					} else {
						layer3_orig[i][j][k] = 0;

					}
		layer3_perExpert.put(-1, layer3_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer2 = layer2_perExpert.get(expertId);

				double[][][] layer3 = new double[24][24][4];
				for (int i = 0; i < 24; i++)
					for (int j = 0; j < 24; j++)
						for (int k = 0; k < 4; k++)
							if (layer2[i][j][k] > 0) {
								layer3[i][j][k] = layer2[i][j][k];

							} else {
								layer3[i][j][k] = 0;

							}

				layer3_perExpert.put(expertId, layer3);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer2 = layer2_perExpert.get(expertId);

					double[][][] layer3 = new double[24][24][4];
					for (int i = 0; i < 24; i++)
						for (int j = 0; j < 24; j++)
							for (int k = 0; k < 4; k++)
								if (layer2[i][j][k] > 0) {
									layer3[i][j][k] = layer2[i][j][k];

								} else {
									layer3[i][j][k] = 0;

								}

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
		double[][][] layer4_orig = new double[12][12][4];
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				for (int k = 0; k < 4; k++) {
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

				double[][][] layer4 = new double[12][12][4];
				for (int i = 0; i < 12; i++)
					for (int j = 0; j < 12; j++)
						for (int k = 0; k < 4; k++) {
							layer4[i][j][k] = 0;
							for (int I = i * 2; I < (i + 1) * 2; I++)
								for (int J = j * 2; J < (j + 1) * 2; J++)
									if (layer3[I][J][k] > layer4[i][j][k])
										layer4[i][j][k] = layer3[I][J][k];
						}

				layer4_perExpert.put(expertId, layer4);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer3 = layer3_perExpert.get(expertId);

					double[][][] layer4 = new double[12][12][4];
					for (int i = 0; i < 12; i++)
						for (int j = 0; j < 12; j++)
							for (int k = 0; k < 4; k++) {
								layer4[i][j][k] = 0;
								for (int I = i * 2; I < (i + 1) * 2; I++)
									for (int J = j * 2; J < (j + 1) * 2; J++)
										if (layer3[I][J][k] > layer4[i][j][k])
											layer4[i][j][k] = layer3[I][J][k];
							}

					layer4_perExpert.put(expertId, layer4);
				}
			}
		}

		return layer4_perExpert;
	}

	Map<Integer, double[]> layer_5(Map<Integer, double[][][]> layer4_perExpert, int repairedLayerId, int[] expertIDs,
			boolean optimized) {
		final int CURRENT_LAYER = 5;
		Map<Integer, double[]> layer5_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[][][] layer4_orig = layer4_perExpert.get(-1);
		double[] layer5_orig = new double[576];
		for (int i = 0; i < 576; i++) {
			int d0 = i / 48;
			int d1 = (i % 48) / 4;
			int d2 = i - d0 * 48 - d1 * 4;
			layer5_orig[i] = layer4_orig[d0][d1][d2];
		}
		layer5_perExpert.put(-1, layer5_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[][][] layer4 = layer4_perExpert.get(expertId);

				double[] layer5 = new double[576];
				for (int i = 0; i < 576; i++) {
					int d0 = i / 48;
					int d1 = (i % 48) / 4;
					int d2 = i - d0 * 48 - d1 * 4;
					layer5[i] = layer4[d0][d1][d2];
				}

				layer5_perExpert.put(expertId, layer5);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[][][] layer4 = layer4_perExpert.get(expertId);

					double[] layer5 = new double[576];
					for (int i = 0; i < 576; i++) {
						int d0 = i / 48;
						int d1 = (i % 48) / 4;
						int d2 = i - d0 * 48 - d1 * 4;
						layer5[i] = layer4[d0][d1][d2];
					}

					layer5_perExpert.put(expertId, layer5);
				}
			}
		}

		return layer5_perExpert;
	}

	Map<Integer, double[]> layer_6(Map<Integer, double[]> layer5_perExpert, int repairedLayerId, int[] expertIDs,
			boolean optimized) {
		final int CURRENT_LAYER = 6;
		Map<Integer, double[]> layer6_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer5_orig = layer5_perExpert.get(-1);
		double[] layer6_orig = new double[128];
		for (int i = 0; i < 128; i++) {
			layer6_orig[i] = internal.biases6[i];
			for (int I = 0; I < 576; I++)
				layer6_orig[i] += internal.weights6[I][i] * layer5_orig[I];
		}
		layer6_perExpert.put(-1, layer6_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][] delta_layer6_perExpert = (double[][][]) weight_delta;

			for (int expertId : expertIDs) {
				double[] layer6 = new double[128];
				for (int i = 0; i < 128; i++) {
					layer6[i] = internal.biases6[i];
					for (int I = 0; I < 576; I++)
						layer6[i] += (internal.weights6[I][i] + delta_layer6_perExpert[expertId][I][i])
								* layer5_orig[I];
				}

				layer6_perExpert.put(expertId, layer6);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer6 = new double[128];
					for (int i = 0; i < 128; i++) {
						layer6[i] = internal.biases6[i];
						for (int I = 0; I < 576; I++)
							layer6[i] += (internal.weights6[I][i] + delta_layer6_perExpert[expertId][I][i])
									* layer5_orig[I];
					}

					layer6_perExpert.put(expertId, layer6);
				}
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId : expertIDs) {
				double[] layer5 = layer5_perExpert.get(expertId);

				double[] layer6 = new double[128];
				for (int i = 0; i < 128; i++) {
					layer6[i] = internal.biases6[i];
					for (int I = 0; I < 576; I++)
						layer6[i] += internal.weights6[I][i] * layer5[I];
				}

				layer6_perExpert.put(expertId, layer6);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer5 = layer5_perExpert.get(expertId);

					double[] layer6 = new double[128];
					for (int i = 0; i < 128; i++) {
						layer6[i] = internal.biases6[i];
						for (int I = 0; I < 576; I++)
							layer6[i] += internal.weights6[I][i] * layer5[I];
					}

					layer6_perExpert.put(expertId, layer6);
				}
			}
		}

		return layer6_perExpert;
	}

	Map<Integer, double[]> layer_7(Map<Integer, double[]> layer6_perExpert, int repairedLayerId, int[] expertIDs,
			boolean optimized) {
		final int CURRENT_LAYER = 7;
		Map<Integer, double[]> layer7_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer6_orig = layer6_perExpert.get(-1);
		double[] layer7_orig = new double[128];
		for (int i = 0; i < 128; i++)
			if (layer6_orig[i] > 0)
				layer7_orig[i] = layer6_orig[i];
			else
				layer7_orig[i] = 0;
		layer7_perExpert.put(-1, layer7_orig);

		if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert. Note: the repair cannot happen in this layer because
			 * no weights are involved.
			 */

			for (int expertId : expertIDs) {
				double[] layer6 = layer6_perExpert.get(expertId);

				double[] layer7 = new double[128];
				for (int i = 0; i < 128; i++)
					if (layer6[i] > 0)
						layer7[i] = layer6[i];
					else
						layer7[i] = 0;

				layer7_perExpert.put(expertId, layer7);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer6 = layer6_perExpert.get(expertId);

					double[] layer7 = new double[128];
					for (int i = 0; i < 128; i++)
						if (layer6[i] > 0)
							layer7[i] = layer6[i];
						else
							layer7[i] = 0;

					layer7_perExpert.put(expertId, layer7);
				}
			}
		}

		return layer7_perExpert;
	}

	Map<Integer, double[]> layer_8(Map<Integer, double[]> layer7_perExpert, int repairedLayerId, int[] expertIDs,
			boolean optimized) {
		final int CURRENT_LAYER = 8;
		Map<Integer, double[]> layer8_perExpert = new HashMap<>();

		/*
		 * Store the original calculation as dummy expert on position -1.
		 */
		double[] layer7_orig = layer7_perExpert.get(-1);
		double[] layer8_orig = new double[10];
		for (int i = 0; i < 10; i++) {
			layer8_orig[i] = internal.biases8[i];
			for (int I = 0; I < 128; I++)
				layer8_orig[i] += internal.weights8[I][i] * layer7_orig[I];
		}
		layer8_perExpert.put(-1, layer8_orig);

		if (repairedLayerId == CURRENT_LAYER) {

			/*
			 * If the repair starts in this layer, then the layer will calculate the output
			 * with the adjusted weights for each expert.
			 */

			double[][][] delta_layer8_perExpert = (double[][][]) weight_delta;

			for (int expertId : expertIDs) {
				double[] layer8 = new double[10];
				for (int i = 0; i < 10; i++) {
					layer8[i] = internal.biases8[i];
					for (int I = 0; I < 128; I++)
						layer8[i] += (internal.weights8[I][i] + delta_layer8_perExpert[expertId][I][i])
								* layer7_orig[I];
				}

				layer8_perExpert.put(expertId, layer8);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer8 = new double[10];
					for (int i = 0; i < 10; i++) {
						layer8[i] = internal.biases8[i];
						for (int I = 0; I < 128; I++)
							layer8[i] += (internal.weights8[I][i] + delta_layer8_perExpert[expertId][I][i])
									* layer7_orig[I];
					}

					layer8_perExpert.put(expertId, layer8);
				}
			}

		} else if (repairedLayerId < CURRENT_LAYER) {

			/*
			 * If the repair happened before this layer, then the layer will calculate the
			 * output for each expert.
			 */

			for (int expertId : expertIDs) {
				double[] layer7 = layer7_perExpert.get(expertId);

				double[] layer8 = new double[10];
				for (int i = 0; i < 10; i++) {
					layer8[i] = internal.biases8[i];
					for (int I = 0; I < 128; I++)
						layer8[i] += internal.weights8[I][i] * layer7[I];
				}

				layer8_perExpert.put(expertId, layer8);
			}

			if (!optimized) {
				// Additional calculation for Average and Full repair.
				for (int expertId = NUMBER_OF_EXPERTS; expertId < NUMBER_OF_EXPERTS + 2; expertId++) {
					double[] layer7 = layer7_perExpert.get(expertId);

					double[] layer8 = new double[10];
					for (int i = 0; i < 10; i++) {
						layer8[i] = internal.biases8[i];
						for (int I = 0; I < 128; I++)
							layer8[i] += internal.weights8[I][i] * layer7[I];
					}

					layer8_perExpert.put(expertId, layer8);
				}
			}
		}

		return layer8_perExpert;
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

		// layer 5: flatten_1
		Map<Integer, double[]> layer5_perExpert = layer_5(layer4_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 6: dense_1
		Map<Integer, double[]> layer6_perExpert = layer_6(layer5_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 7: activation_3
		Map<Integer, double[]> layer7_perExpert = layer_7(layer6_perExpert, repairedLayerId, expertIDs, optimized);

		// layer 8: dense_2
		Map<Integer, double[]> layer8_perExpert = layer_8(layer7_perExpert, repairedLayerId, expertIDs, optimized);

		/*
		 * At this point we have layer8_perExpert with the original calculation results
		 * on position -1, and if there been a repaired layer, then the expert results
		 * are stored in positions 0-9.
		 */

		return layer8_perExpert;
	}

}

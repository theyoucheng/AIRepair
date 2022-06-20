public class CIFAR10_DNNt_Original {

	private CIFAR10_InternalData internal;

	// weights0: shape is 3x3x3x32
	// biases0: shape is 32
	// weights2: shape is 3x3x32x32
	// biases2: shape is 32
	// weights5: shape is 3x3x32x64
	// biases5: shape is 64
	// weights7: shape is 3x3x64x64
	// biases7: shape is 64
	// weights11: shape is 1600
	// biases11: shape is 512
	// weights13: shape is 512
	// biases13: shape is 10

	public CIFAR10_DNNt_Original(CIFAR10_InternalData internal) {
		this.internal = internal;
	}

	// the DNN input is of shap 32x32x3
	int run(double[][][] input) {

		// layer 0: conv2d_1
		double[][][] layer0 = new double[30][30][32];
		for (int i = 0; i < 30; i++)
			for (int j = 0; j < 30; j++)
				for (int k = 0; k < 32; k++) {
					layer0[i][j][k] = internal.biases0[k];
					for (int I = 0; I < 3; I++)
						for (int J = 0; J < 3; J++)
							for (int K = 0; K < 3; K++)
								layer0[i][j][k] += internal.weights0[I][J][K][k] * input[i + I][j + J][K];
				}

		// layer 1: activation_1
		double[][][] layer1 = new double[30][30][32];
		for (int i = 0; i < 30; i++)
			for (int j = 0; j < 30; j++)
				for (int k = 0; k < 32; k++)
					if (layer0[i][j][k] > 0)
						layer1[i][j][k] = layer0[i][j][k];
					else
						layer1[i][j][k] = 0;

		// layer 2: conv2d_2
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

		// layer 3: activation_2
		double[][][] layer3 = new double[28][28][32];
		for (int i = 0; i < 28; i++)
			for (int j = 0; j < 28; j++)
				for (int k = 0; k < 32; k++)
					if (layer2[i][j][k] > 0)
						layer3[i][j][k] = layer2[i][j][k];
					else
						layer3[i][j][k] = 0;

		// layer 4: max_pooling2d_1
		double[][][] layer4 = new double[14][14][32];
		for (int i = 0; i < 14; i++)
			for (int j = 0; j < 14; j++)
				for (int k = 0; k < 32; k++) {
					layer4[i][j][k] = 0;
					for (int I = i * 2; I < (i + 1) * 2; I++)
						for (int J = j * 2; J < (j + 1) * 2; J++)
							if (layer3[I][J][k] > layer4[i][j][k])
								layer4[i][j][k] = layer3[I][J][k];
				}

		// layer 5: conv2d_3
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

		// layer 6: activation_3
		double[][][] layer6 = new double[12][12][64];
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				for (int k = 0; k < 64; k++)
					if (layer5[i][j][k] > 0)
						layer6[i][j][k] = layer5[i][j][k];
					else
						layer6[i][j][k] = 0;

		// layer 7: conv2d_4
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

		// layer 8: activation_4
		double[][][] layer8 = new double[10][10][64];
		for (int i = 0; i < 10; i++)
			for (int j = 0; j < 10; j++)
				for (int k = 0; k < 64; k++)
					if (layer7[i][j][k] > 0)
						layer8[i][j][k] = layer7[i][j][k];
					else
						layer8[i][j][k] = 0;

		// layer 9: max_pooling2d_2
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

		// layer 10: flatten_1
		double[] layer10 = new double[1600];
		for (int i = 0; i < 1600; i++) {
			int d0 = i / 320;
			int d1 = (i % 320) / 64;
			int d2 = i - d0 * 320 - d1 * 64;
			layer10[i] = layer9[d0][d1][d2];
		}

		// layer 11: dense_1
		double[] layer11 = new double[512];
		for (int i = 0; i < 512; i++) {
			layer11[i] = internal.biases11[i];
			for (int I = 0; I < 1600; I++)
				layer11[i] += internal.weights11[I][i] * layer10[I];
		}

		// layer 12: activation_5
		double[] layer12 = new double[512];
		for (int i = 0; i < 512; i++)
			if (layer11[i] > 0)
				layer12[i] = layer11[i];
			else
				layer12[i] = 0;

		// layer 13: dense_2
		double[] layer13 = new double[10];
		for (int i = 0; i < 10; i++) {
			layer13[i] = internal.biases13[i];
			for (int I = 0; I < 512; I++)
				layer13[i] += internal.weights13[I][i] * layer12[I];
		}

		// layer 14: activation_6
		double[] layer14 = new double[10];
		for (int i = 0; i < 10; i++)
			layer14[i] = layer13[i]; // alala
		int ret = 0;
		double res = -100000;
		for (int i = 0; i < 10; i++) {
			if (layer14[i] > res) {
				res = layer14[i];
				ret = i;
			}
		}
		return ret;
	}

}

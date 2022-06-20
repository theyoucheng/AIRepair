import java.io.*;

public class CIFAR10_InternalData {

	public Double[][][][] weights0;
	public Double[][][][] weights2;
	public Double[][][][] weights5;
	public Double[][][][] weights7;
	public Double[][] weights11;
	public Double[][] weights13;
	
	public Double[] biases0;
	public Double[] biases2;
	public Double[] biases5;
	public Double[] biases7;
	public Double[] biases11;
	public Double[] biases13;

	public CIFAR10_InternalData(String path, String weights0file, String weights2file, String weights5file, String weights7file,
			String weights11file, String weights13file, String bias0file, String bias2file, String bias5file,
			String bias7file, String bias11file, String bias13file) throws NumberFormatException, IOException {

		int index = 0;
		Double[] Wvalues = null;
		Double[] Bvalues = null;
		File file = null;
		BufferedReader br = null;
		String st = null;

		file = new File(path + "/" + weights0file);
		br = new BufferedReader(new FileReader(file));
		Wvalues = new Double[864];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			String[] vals = st.split(",");
			Wvalues[index] = Double.valueOf(vals[0]);
			index++;
			Wvalues[index] = Double.valueOf(vals[1]);
			index++;
			Wvalues[index] = Double.valueOf(vals[2]);
			index++;
			Wvalues[index] = Double.valueOf(vals[3]);
			index++;
			Wvalues[index] = Double.valueOf(vals[4]);
			index++;
			Wvalues[index] = Double.valueOf(vals[5]);
			index++;
			Wvalues[index] = Double.valueOf(vals[6]);
			index++;
			Wvalues[index] = Double.valueOf(vals[7]);
			index++;
			Wvalues[index] = Double.valueOf(vals[8]);
			index++;
			Wvalues[index] = Double.valueOf(vals[9]);
			index++;
			Wvalues[index] = Double.valueOf(vals[10]);
			index++;
			Wvalues[index] = Double.valueOf(vals[11]);
			index++;
			Wvalues[index] = Double.valueOf(vals[12]);
			index++;
			Wvalues[index] = Double.valueOf(vals[13]);
			index++;
			Wvalues[index] = Double.valueOf(vals[14]);
			index++;
			Wvalues[index] = Double.valueOf(vals[15]);
			index++;
			Wvalues[index] = Double.valueOf(vals[16]);
			index++;
			Wvalues[index] = Double.valueOf(vals[17]);
			index++;
			Wvalues[index] = Double.valueOf(vals[18]);
			index++;
			Wvalues[index] = Double.valueOf(vals[19]);
			index++;
			Wvalues[index] = Double.valueOf(vals[20]);
			index++;
			Wvalues[index] = Double.valueOf(vals[21]);
			index++;
			Wvalues[index] = Double.valueOf(vals[22]);
			index++;
			Wvalues[index] = Double.valueOf(vals[23]);
			index++;
			Wvalues[index] = Double.valueOf(vals[24]);
			index++;
			Wvalues[index] = Double.valueOf(vals[25]);
			index++;
			Wvalues[index] = Double.valueOf(vals[26]);
			index++;
			Wvalues[index] = Double.valueOf(vals[27]);
			index++;
			Wvalues[index] = Double.valueOf(vals[28]);
			index++;
			Wvalues[index] = Double.valueOf(vals[29]);
			index++;
			Wvalues[index] = Double.valueOf(vals[30]);
			index++;
			Wvalues[index] = Double.valueOf(vals[31]);
			index++;
		}
		br.close();
		file = new File(path + "/" + bias0file);
		br = new BufferedReader(new FileReader(file));
		Bvalues = new Double[32];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			Bvalues[index] = Double.valueOf(st);
			index++;
		}
		biases0 = new Double[32];
		index = 0;
		for (int k = 0; k < 32; k++) {
			biases0[k] = Bvalues[index];
			index++;
		}
		weights0 = new Double[3][3][3][32];
		index = 0;
		for (int I = 0; I < 3; I++)
			for (int J = 0; J < 3; J++)
				for (int K = 0; K < 3; K++)
					for (int k = 0; k < 32; k++) {
						weights0[I][J][K][k] = Wvalues[index];
						index++;
					}
		br.close();

		file = new File(path + "/" + weights2file);
		br = new BufferedReader(new FileReader(file));
		Wvalues = new Double[9216];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			String[] vals = st.split(",");
			Wvalues[index] = Double.valueOf(vals[0]);
			index++;
			Wvalues[index] = Double.valueOf(vals[1]);
			index++;
			Wvalues[index] = Double.valueOf(vals[2]);
			index++;
			Wvalues[index] = Double.valueOf(vals[3]);
			index++;
			Wvalues[index] = Double.valueOf(vals[4]);
			index++;
			Wvalues[index] = Double.valueOf(vals[5]);
			index++;
			Wvalues[index] = Double.valueOf(vals[6]);
			index++;
			Wvalues[index] = Double.valueOf(vals[7]);
			index++;
			Wvalues[index] = Double.valueOf(vals[8]);
			index++;
			Wvalues[index] = Double.valueOf(vals[9]);
			index++;
			Wvalues[index] = Double.valueOf(vals[10]);
			index++;
			Wvalues[index] = Double.valueOf(vals[11]);
			index++;
			Wvalues[index] = Double.valueOf(vals[12]);
			index++;
			Wvalues[index] = Double.valueOf(vals[13]);
			index++;
			Wvalues[index] = Double.valueOf(vals[14]);
			index++;
			Wvalues[index] = Double.valueOf(vals[15]);
			index++;
			Wvalues[index] = Double.valueOf(vals[16]);
			index++;
			Wvalues[index] = Double.valueOf(vals[17]);
			index++;
			Wvalues[index] = Double.valueOf(vals[18]);
			index++;
			Wvalues[index] = Double.valueOf(vals[19]);
			index++;
			Wvalues[index] = Double.valueOf(vals[20]);
			index++;
			Wvalues[index] = Double.valueOf(vals[21]);
			index++;
			Wvalues[index] = Double.valueOf(vals[22]);
			index++;
			Wvalues[index] = Double.valueOf(vals[23]);
			index++;
			Wvalues[index] = Double.valueOf(vals[24]);
			index++;
			Wvalues[index] = Double.valueOf(vals[25]);
			index++;
			Wvalues[index] = Double.valueOf(vals[26]);
			index++;
			Wvalues[index] = Double.valueOf(vals[27]);
			index++;
			Wvalues[index] = Double.valueOf(vals[28]);
			index++;
			Wvalues[index] = Double.valueOf(vals[29]);
			index++;
			Wvalues[index] = Double.valueOf(vals[30]);
			index++;
			Wvalues[index] = Double.valueOf(vals[31]);
			index++;
		}
		br.close();
		file = new File(path + "/" + bias2file);
		br = new BufferedReader(new FileReader(file));
		Bvalues = new Double[32];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			Bvalues[index] = Double.valueOf(st);
			index++;
		}
		biases2 = new Double[32];
		index = 0;
		for (int k = 0; k < 32; k++) {
			biases2[k] = Bvalues[index];
			index++;
		}
		weights2 = new Double[3][3][32][32];
		index = 0;
		for (int I = 0; I < 3; I++)
			for (int J = 0; J < 3; J++)
				for (int K = 0; K < 32; K++)
					for (int k = 0; k < 32; k++) {
						weights2[I][J][K][k] = Wvalues[index];
						index++;
					}
		br.close();

		file = new File(path + "/" + weights5file);
		br = new BufferedReader(new FileReader(file));
		Wvalues = new Double[18432];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			String[] vals = st.split(",");
			Wvalues[index] = Double.valueOf(vals[0]);
			index++;
			Wvalues[index] = Double.valueOf(vals[1]);
			index++;
			Wvalues[index] = Double.valueOf(vals[2]);
			index++;
			Wvalues[index] = Double.valueOf(vals[3]);
			index++;
			Wvalues[index] = Double.valueOf(vals[4]);
			index++;
			Wvalues[index] = Double.valueOf(vals[5]);
			index++;
			Wvalues[index] = Double.valueOf(vals[6]);
			index++;
			Wvalues[index] = Double.valueOf(vals[7]);
			index++;
			Wvalues[index] = Double.valueOf(vals[8]);
			index++;
			Wvalues[index] = Double.valueOf(vals[9]);
			index++;
			Wvalues[index] = Double.valueOf(vals[10]);
			index++;
			Wvalues[index] = Double.valueOf(vals[11]);
			index++;
			Wvalues[index] = Double.valueOf(vals[12]);
			index++;
			Wvalues[index] = Double.valueOf(vals[13]);
			index++;
			Wvalues[index] = Double.valueOf(vals[14]);
			index++;
			Wvalues[index] = Double.valueOf(vals[15]);
			index++;
			Wvalues[index] = Double.valueOf(vals[16]);
			index++;
			Wvalues[index] = Double.valueOf(vals[17]);
			index++;
			Wvalues[index] = Double.valueOf(vals[18]);
			index++;
			Wvalues[index] = Double.valueOf(vals[19]);
			index++;
			Wvalues[index] = Double.valueOf(vals[20]);
			index++;
			Wvalues[index] = Double.valueOf(vals[21]);
			index++;
			Wvalues[index] = Double.valueOf(vals[22]);
			index++;
			Wvalues[index] = Double.valueOf(vals[23]);
			index++;
			Wvalues[index] = Double.valueOf(vals[24]);
			index++;
			Wvalues[index] = Double.valueOf(vals[25]);
			index++;
			Wvalues[index] = Double.valueOf(vals[26]);
			index++;
			Wvalues[index] = Double.valueOf(vals[27]);
			index++;
			Wvalues[index] = Double.valueOf(vals[28]);
			index++;
			Wvalues[index] = Double.valueOf(vals[29]);
			index++;
			Wvalues[index] = Double.valueOf(vals[30]);
			index++;
			Wvalues[index] = Double.valueOf(vals[31]);
			index++;
			Wvalues[index] = Double.valueOf(vals[32]);
			index++;
			Wvalues[index] = Double.valueOf(vals[33]);
			index++;
			Wvalues[index] = Double.valueOf(vals[34]);
			index++;
			Wvalues[index] = Double.valueOf(vals[35]);
			index++;
			Wvalues[index] = Double.valueOf(vals[36]);
			index++;
			Wvalues[index] = Double.valueOf(vals[37]);
			index++;
			Wvalues[index] = Double.valueOf(vals[38]);
			index++;
			Wvalues[index] = Double.valueOf(vals[39]);
			index++;
			Wvalues[index] = Double.valueOf(vals[40]);
			index++;
			Wvalues[index] = Double.valueOf(vals[41]);
			index++;
			Wvalues[index] = Double.valueOf(vals[42]);
			index++;
			Wvalues[index] = Double.valueOf(vals[43]);
			index++;
			Wvalues[index] = Double.valueOf(vals[44]);
			index++;
			Wvalues[index] = Double.valueOf(vals[45]);
			index++;
			Wvalues[index] = Double.valueOf(vals[46]);
			index++;
			Wvalues[index] = Double.valueOf(vals[47]);
			index++;
			Wvalues[index] = Double.valueOf(vals[48]);
			index++;
			Wvalues[index] = Double.valueOf(vals[49]);
			index++;
			Wvalues[index] = Double.valueOf(vals[50]);
			index++;
			Wvalues[index] = Double.valueOf(vals[51]);
			index++;
			Wvalues[index] = Double.valueOf(vals[52]);
			index++;
			Wvalues[index] = Double.valueOf(vals[53]);
			index++;
			Wvalues[index] = Double.valueOf(vals[54]);
			index++;
			Wvalues[index] = Double.valueOf(vals[55]);
			index++;
			Wvalues[index] = Double.valueOf(vals[56]);
			index++;
			Wvalues[index] = Double.valueOf(vals[57]);
			index++;
			Wvalues[index] = Double.valueOf(vals[58]);
			index++;
			Wvalues[index] = Double.valueOf(vals[59]);
			index++;
			Wvalues[index] = Double.valueOf(vals[60]);
			index++;
			Wvalues[index] = Double.valueOf(vals[61]);
			index++;
			Wvalues[index] = Double.valueOf(vals[62]);
			index++;
			Wvalues[index] = Double.valueOf(vals[63]);
			index++;
		}
		br.close();
		file = new File(path + "/" + bias5file);
		br = new BufferedReader(new FileReader(file));
		Bvalues = new Double[64];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			Bvalues[index] = Double.valueOf(st);
			index++;
		}
		biases5 = new Double[64];
		index = 0;
		for (int k = 0; k < 64; k++) {
			biases5[k] = Bvalues[index];
			index++;
		}
		weights5 = new Double[3][3][32][64];
		index = 0;
		for (int I = 0; I < 3; I++)
			for (int J = 0; J < 3; J++)
				for (int K = 0; K < 32; K++)
					for (int k = 0; k < 64; k++) {
						weights5[I][J][K][k] = Wvalues[index];
						index++;
					}
		br.close();

		file = new File(path + "/" + weights7file);
		br = new BufferedReader(new FileReader(file));
		Wvalues = new Double[36864];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			String[] vals = st.split(",");
			Wvalues[index] = Double.valueOf(vals[0]);
			index++;
			Wvalues[index] = Double.valueOf(vals[1]);
			index++;
			Wvalues[index] = Double.valueOf(vals[2]);
			index++;
			Wvalues[index] = Double.valueOf(vals[3]);
			index++;
			Wvalues[index] = Double.valueOf(vals[4]);
			index++;
			Wvalues[index] = Double.valueOf(vals[5]);
			index++;
			Wvalues[index] = Double.valueOf(vals[6]);
			index++;
			Wvalues[index] = Double.valueOf(vals[7]);
			index++;
			Wvalues[index] = Double.valueOf(vals[8]);
			index++;
			Wvalues[index] = Double.valueOf(vals[9]);
			index++;
			Wvalues[index] = Double.valueOf(vals[10]);
			index++;
			Wvalues[index] = Double.valueOf(vals[11]);
			index++;
			Wvalues[index] = Double.valueOf(vals[12]);
			index++;
			Wvalues[index] = Double.valueOf(vals[13]);
			index++;
			Wvalues[index] = Double.valueOf(vals[14]);
			index++;
			Wvalues[index] = Double.valueOf(vals[15]);
			index++;
			Wvalues[index] = Double.valueOf(vals[16]);
			index++;
			Wvalues[index] = Double.valueOf(vals[17]);
			index++;
			Wvalues[index] = Double.valueOf(vals[18]);
			index++;
			Wvalues[index] = Double.valueOf(vals[19]);
			index++;
			Wvalues[index] = Double.valueOf(vals[20]);
			index++;
			Wvalues[index] = Double.valueOf(vals[21]);
			index++;
			Wvalues[index] = Double.valueOf(vals[22]);
			index++;
			Wvalues[index] = Double.valueOf(vals[23]);
			index++;
			Wvalues[index] = Double.valueOf(vals[24]);
			index++;
			Wvalues[index] = Double.valueOf(vals[25]);
			index++;
			Wvalues[index] = Double.valueOf(vals[26]);
			index++;
			Wvalues[index] = Double.valueOf(vals[27]);
			index++;
			Wvalues[index] = Double.valueOf(vals[28]);
			index++;
			Wvalues[index] = Double.valueOf(vals[29]);
			index++;
			Wvalues[index] = Double.valueOf(vals[30]);
			index++;
			Wvalues[index] = Double.valueOf(vals[31]);
			index++;
			Wvalues[index] = Double.valueOf(vals[32]);
			index++;
			Wvalues[index] = Double.valueOf(vals[33]);
			index++;
			Wvalues[index] = Double.valueOf(vals[34]);
			index++;
			Wvalues[index] = Double.valueOf(vals[35]);
			index++;
			Wvalues[index] = Double.valueOf(vals[36]);
			index++;
			Wvalues[index] = Double.valueOf(vals[37]);
			index++;
			Wvalues[index] = Double.valueOf(vals[38]);
			index++;
			Wvalues[index] = Double.valueOf(vals[39]);
			index++;
			Wvalues[index] = Double.valueOf(vals[40]);
			index++;
			Wvalues[index] = Double.valueOf(vals[41]);
			index++;
			Wvalues[index] = Double.valueOf(vals[42]);
			index++;
			Wvalues[index] = Double.valueOf(vals[43]);
			index++;
			Wvalues[index] = Double.valueOf(vals[44]);
			index++;
			Wvalues[index] = Double.valueOf(vals[45]);
			index++;
			Wvalues[index] = Double.valueOf(vals[46]);
			index++;
			Wvalues[index] = Double.valueOf(vals[47]);
			index++;
			Wvalues[index] = Double.valueOf(vals[48]);
			index++;
			Wvalues[index] = Double.valueOf(vals[49]);
			index++;
			Wvalues[index] = Double.valueOf(vals[50]);
			index++;
			Wvalues[index] = Double.valueOf(vals[51]);
			index++;
			Wvalues[index] = Double.valueOf(vals[52]);
			index++;
			Wvalues[index] = Double.valueOf(vals[53]);
			index++;
			Wvalues[index] = Double.valueOf(vals[54]);
			index++;
			Wvalues[index] = Double.valueOf(vals[55]);
			index++;
			Wvalues[index] = Double.valueOf(vals[56]);
			index++;
			Wvalues[index] = Double.valueOf(vals[57]);
			index++;
			Wvalues[index] = Double.valueOf(vals[58]);
			index++;
			Wvalues[index] = Double.valueOf(vals[59]);
			index++;
			Wvalues[index] = Double.valueOf(vals[60]);
			index++;
			Wvalues[index] = Double.valueOf(vals[61]);
			index++;
			Wvalues[index] = Double.valueOf(vals[62]);
			index++;
			Wvalues[index] = Double.valueOf(vals[63]);
			index++;
		}
		br.close();
		file = new File(path + "/" + bias7file);
		br = new BufferedReader(new FileReader(file));
		Bvalues = new Double[64];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			Bvalues[index] = Double.valueOf(st);
			index++;
		}
		biases7 = new Double[64];
		index = 0;
		for (int k = 0; k < 64; k++) {
			biases7[k] = Bvalues[index];
			index++;
		}
		weights7 = new Double[3][3][64][64];
		index = 0;
		for (int I = 0; I < 3; I++)
			for (int J = 0; J < 3; J++)
				for (int K = 0; K < 64; K++)
					for (int k = 0; k < 64; k++) {
						weights7[I][J][K][k] = Wvalues[index];
						index++;
					}
		br.close();

		file = new File(path + "/" + weights11file);
		br = new BufferedReader(new FileReader(file));
		Wvalues = new Double[819200];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			String[] vals = st.split(",");
			for (int i = 0; i < 512; i++) {
				Wvalues[index] = Double.valueOf(vals[i]);
				index++;
			}
		}
		br.close();
		file = new File(path + "/" + bias11file);
		br = new BufferedReader(new FileReader(file));
		Bvalues = new Double[512];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			Bvalues[index] = Double.valueOf(st);
			index++;
		}
		biases11 = new Double[512];
		index = 0;
		for (int k = 0; k < 512; k++) {
			biases11[k] = Bvalues[index];
			index++;
		}
		weights11 = new Double[1600][512];
		index = 0;
		for (int I = 0; I < 1600; I++)
			for (int J = 0; J < 512; J++) {
				weights11[I][J] = Wvalues[index];
				index++;
			}
		br.close();

		file = new File(path + "/" + weights13file);
		br = new BufferedReader(new FileReader(file));
		Wvalues = new Double[5120];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			String[] vals = st.split(",");
			for (int i = 0; i < 10; i++) {
				Wvalues[index] = Double.valueOf(vals[i]);
				index++;
			}
		}
		br.close();
		file = new File(path + "/" + bias13file);
		br = new BufferedReader(new FileReader(file));
		Bvalues = new Double[10];
		index = 0;
		while ((st = br.readLine()) != null) {
			if (st.isEmpty())
				continue;
			Bvalues[index] = Double.valueOf(st);
			index++;
		}
		biases13 = new Double[10];
		index = 0;
		for (int k = 0; k < 10; k++) {
			biases13[k] = Bvalues[index];
			index++;
		}
		weights13 = new Double[512][10];
		index = 0;
		for (int I = 0; I < 512; I++)
			for (int J = 0; J < 10; J++) {
				weights13[I][J] = Wvalues[index];
				index++;
			}
		br.close();

	}
}
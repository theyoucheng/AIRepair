import java.io.*;
public class InternalData {
  public Double[][][][] weights0;
  public Double[][][][] weights2;
  public Double[][] weights6;
  public Double[][] weights8;
  public Double[] biases0;
  public Double[] biases2;
  public Double[] biases6;
  public Double[] biases8;

  public InternalData(String weights0file,String weights2file,String weights6file,String weights8file,String bias0file,String bias2file,String bias6file,String bias8file) throws NumberFormatException, IOException {

    String path = "./params/";
    int index = 0;
    Double[] Wvalues = null;
    Double[] Bvalues = null;
    File file = null;
    BufferedReader br = null;
    String st = null;

    file = new File(path + weights0file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[18];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        Wvalues[index] = Double.valueOf(vals[0]);
        index++;
        Wvalues[index] = Double.valueOf(vals[1]);
        index++;
    }
    br.close();
    file = new File(path + bias0file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[2];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases0 = new Double[2];
    index = 0;
    for (int k = 0; k < 2; k++) {
      biases0[k] = Bvalues[index];
      index++;
    }
    weights0 = new Double[3][3][1][2];
    index = 0;
    for (int I = 0; I < 3; I++)
      for (int J = 0; J < 3; J++)
        for (int K = 0; K < 1; K++)
          for (int k = 0; k < 2; k++)
          {
            weights0[I][J][K][k] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights2file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[72];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        Wvalues[index] = Double.valueOf(vals[0]);
        index++;
        Wvalues[index] = Double.valueOf(vals[1]);
        index++;
        Wvalues[index] = Double.valueOf(vals[2]);
        index++;
        Wvalues[index] = Double.valueOf(vals[3]);
        index++;
    }
    br.close();
    file = new File(path + bias2file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[4];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases2 = new Double[4];
    index = 0;
    for (int k = 0; k < 4; k++) {
      biases2[k] = Bvalues[index];
      index++;
    }
    weights2 = new Double[3][3][2][4];
    index = 0;
    for (int I = 0; I < 3; I++)
      for (int J = 0; J < 3; J++)
        for (int K = 0; K < 2; K++)
          for (int k = 0; k < 4; k++)
          {
            weights2[I][J][K][k] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights6file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[73728];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 128; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias6file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[128];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases6 = new Double[128];
    index = 0;
    for (int k = 0; k < 128; k++) {
      biases6[k] = Bvalues[index];
      index++;
    }
    weights6 = new Double[576][128];
    index = 0;
    for (int I = 0; I < 576; I++)
      for (int J = 0; J < 128; J++)
          {
            weights6[I][J] = Wvalues[index];
            index++;
          }
    br.close();


    file = new File(path + weights8file);
    br = new BufferedReader(new FileReader(file));
    Wvalues = new Double[1280];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      String[] vals = st.split(",");
        for (int i = 0; i < 10; i++) {
          Wvalues[index] = Double.valueOf(vals[i]);
          index ++;
        }
    }
    br.close();
    file = new File(path + bias8file);
    br = new BufferedReader(new FileReader(file));
    Bvalues = new Double[10];
    index = 0;
    while ((st = br.readLine()) != null) {
      if (st.isEmpty()) continue;
      Bvalues[index] = Double.valueOf(st);
      index++;
    }
    biases8 = new Double[10];
    index = 0;
    for (int k = 0; k < 10; k++) {
      biases8[k] = Bvalues[index];
      index++;
    }
    weights8 = new Double[128][10];
    index = 0;
    for (int I = 0; I < 128; I++)
      for (int J = 0; J < 10; J++)
          {
            weights8[I][J] = Wvalues[index];
            index++;
          }
    br.close();

  }
}
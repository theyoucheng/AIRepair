import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Run {

    public static void main(String[] args){
		try {
			InternalData data = new InternalData("weights0.txt","weights2.txt","weights5.txt","weights7.txt","weights11.txt","weights13.txt","biases0.txt","biases2.txt","biases5.txt","biases7.txt","biases11.txt","biases13.txt");
			//InternalData data = new InternalData("weights0.txt","weights2.txt","weights5.txt","weights7.txt","weights11.txt","weights13.txt","weights15.txt","biases0.txt","biases2.txt","biases5.txt","biases7.txt","biases11.txt","biases13.txt","biases15.txt");
			//InternalData data = new InternalData("weights1.txt","weights2.txt","weights3.txt","weights4.txt","weights5.txt","weights6.txt","biases1.txt","biases2.txt","biases3.txt","biases4.txt","biases5.txt","biases6.txt");
			//InternalData data = new InternalData("weights1.txt","weights2.txt","weights3.txt","weights4.txt","weights5.txt","biases1.txt","biases2.txt","biases3.txt","biases4.txt","biases5.txt");
			//InternalData data = new InternalData("weights1.txt","weights2.txt","weights3.txt","weights4.txt","biases1.txt","biases2.txt","biases3.txt","biases4.txt");
			
			DNNt model = new DNNt(data);
			
			
      String labelFile = "adv-val-data/cifar10_adv_val_labels.txt";
			//String labelFile = "adv-data/cifar10_adv_labels.txt";
			File file = new File(labelFile); 
	    	BufferedReader br = new BufferedReader(new FileReader(file)); 
	    	String st; 
	    	Integer[] labels = new Integer[50000];
	    	int index = 0;
	    	while ((st = br.readLine()) != null) {
	    		   labels[index] = Integer.valueOf(st);
	    		   index++;
	    	}
	    	
	    	br.close();
			//String inputFile = "./data/mnist_train_csv.txt";
			String inputFile = "adv-val-data/cifar10_adv_val_csv_fgsm_epsilon0.01.txt"; //"./data/cifar_train_csv.txt";
			//String inputFile = "adv-data/cifar10_adv_csv_fgsm_epsilon0.01.txt"; //"./data/cifar_train_csv.txt";
			file = new File(inputFile); 
	    	br = new BufferedReader(new FileReader(file)); 
	    	int count = 0;
	    	int pass = 0;
	    	int fail = 0;
	    	while ((st = br.readLine()) != null) {
	    	    //System.out.println("INPUT:" + st); 
	    	    String[] values = st.split(",");
	    	    double[][][] input = new double[32][32][3];
	    	    index = 0;
	    	    while (index < values.length) {
	    	    	for (int i = 0; i < 32 ; i++)
	    	    		for (int j = 0; j < 32; j++)
	    	    			for (int k = 0; k < 3; k++)
	    	    			{
	    	    				 Double val = Double.valueOf(values[index]);
	    	    				 index++;
                     //input[i][j][k] = (double)(val/255.0);
	    	    	       input[i][j][k] = (val);
	    	    			}
	    	    }
	    	   
	    	    int label = model.run(input);
	    	    
	    	    //System.out.println("MODEL OUTPUT:" + label);
	    	    //System.out.println("ACTUAL OUTPUT:" + labels[count]);
	    	    
	    	    
	    	    if (label == labels[count])
	    	    	pass++;
	    	    else
	    	    	fail++;
	    	    
	    	    count++;

            if (count%100==0) {
	    	      double accuracy = (((double)pass)/(pass+fail))*100.0;
              System.out.println("PASS:"+ pass + "/FAIL:"+fail + "/accuracy:"+ accuracy);
            }
           
	    	    
	    	}
	    	double accuracy = (((double)pass)/50000.0)*100.0;
	    	System.out.println("PASS:"+ pass + "FAIL:"+fail + "accuracy:"+ accuracy);
	    	
	    	br.close();
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}

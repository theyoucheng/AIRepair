
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

//import gov.nasa.jpf.symbc.Debug;

public class MNIST0_DNNt_pattern_based {


	static int MODEL = 0; // 0: low-quality, 1: poisoned, 2: highquality
	static String COMB_METHOD = "NAIVE"; // NAIVE, AVERAGE, FULL, PREC, CONF, VOTES, P+V+C, ORIG
	static Double[] train_precision = new Double[] {0.971,0.9849,0.6278,0.9502,0.677,0.9135,0.99,0.9571,0.9748,0.9312}; // ONLY IF COMB_METHOD = PREC,CONF,VOTES,P+V+C ; precision of the expert on the train set
	
	static String SELECT_EXPERTS = "YES"; // YES, NO - DO WE WANT TO USE ONLY THOSE EXPERTS WHOSE F1 VALUE ON TRAINSET IS MORE THAN ORIGINAL MODEL  
	static Integer[] exp_labs = new Integer[] {6,8,9}; // ONLY IF SELECT_EXPERTS = YES, EXPERTS WHOSE F1 VALUE ON TRAIN SET IS MORE THAN ORIGINAL MODEL
	
	static int LABEL = -1; // 0 to 9 (Targetted), or -1 (All labels together)
	
	
	static String path = null;
	static double label_conf = -1;
    static double train_prec = 0.0;
    static List<Integer> EXPERTS = new ArrayList<Integer>();
   

	private MNIST0_InternalData internal;

	static Double[][] initWeights0 = new Double[576][128];
	static Double[][] initWeights1 = new Double[576][128];
	static Double[][] initWeights2 = new Double[576][128];
	static Double[][] initWeights3 = new Double[576][128];
	static Double[][] initWeights4 = new Double[576][128];
	static Double[][] initWeights5 = new Double[576][128];
	static Double[][] initWeights6 = new Double[576][128];
	static Double[][] initWeights7 = new Double[576][128];
	static Double[][] initWeights8 = new Double[576][128];
	static Double[][] initWeights9 = new Double[576][128];
	static Double[][] initWeights10 = new Double[576][128];
	static Double[][] average_weights6 = new Double[576][128];
	
	

	static ArrayList<Integer> num0 = new ArrayList<Integer>();
	static ArrayList<Integer> num1 = new ArrayList<Integer>();
	static ArrayList<Double>  num2 = new ArrayList<Double>();

	// weights0: shape is 3x3x1x2
	// biases0: shape is 2
	// weights2: shape is 3x3x2x4
	// biases2: shape is 4
	// weights6: shape is 576
	// biases6: shape is 128
	// weights8: shape is 128
	// biases8: shape is 10

	public MNIST0_DNNt_pattern_based(MNIST0_InternalData internal) throws IOException {
		this.internal = internal;

		init(true, this.internal);     


	}
	
	public static void loaddeltas(int lab) throws IOException {
		
		num0.clear();
		num1.clear();
		num2.clear();
		File readfile = null;
		FileReader frread;
		BufferedReader brread;
		String line;
		Pattern p = Pattern.compile("[-+]?[0-9]*\\.?[0-9]+");

		String expert_file = path;
		
		if (lab == 10)
			expert_file = expert_file + "\\full.txt";
		else
			expert_file = expert_file + "\\label" + lab + ".txt";
			
		
		frread = new FileReader(readfile);
		brread = new BufferedReader(frread);
		line="";
		Matcher m;
		int count=0;
		Integer[] edgeslist;
		int keys=0;
		Map<Integer,Integer[]> Edges = new HashMap<Integer,Integer[]>();
		String [] r;
		Integer[] numbers= {};
		brread.readLine();
		brread.readLine();

		while((line = brread.readLine()) != null){
			String a="";
			String b="";
			Double c=0.0;
			int countnumbers=0;
			//   System.out.println("=>"+line);
			if(line.contains("define"))
			{
				line=line.replaceAll("_", " ");
				String[] nums = line.replaceAll("[^0-9 ]", "").trim().split(" +");
				//		   System.out.println(nums[1]);
				num0.add(Integer.valueOf(nums[0]));
				num1.add(Integer.valueOf(nums[1]));
			}
			else if(line.contains("/") && line.chars().filter(ch -> ch == '.').count()==2)			
			{
				m = p.matcher(line);
				while (m.find()) {
					countnumbers++;
					if(countnumbers==1)
					{
						a=m.group();
					}
					else if(countnumbers==2)
					{
						b=m.group();
					}
				}
				c=Double.valueOf(a)/Double.valueOf(b);
				if(line.contains("-"))
				{
					c=c*-1;	
				}
				num2.add(c);
			}
			else if(line.contains("/") && line.chars().filter(ch -> ch == '.').count()==1)			
			{
				m = p.matcher(line);
				while (m.find()) {
					a=m.group();
				}
				if(line.contains("-"))
				{
					c=-1.0;	
				}
				else
				{
					c=1.0;
				}
				line = brread.readLine();
				m = p.matcher(line);
				while (m.find()) {
					b=m.group();
				}
			
				c=c*Double.valueOf(a)/Double.valueOf(b);
			
				num2.add(c);
			}
			else 	
			{
				m = p.matcher(line);
				while (m.find()) {
					c=Double.valueOf(m.group());
				}
				if(line.contains("-"))
				{
					c=c*-1;	
				}
				num2.add(c);
			}
			
		}
	}
	public static void init(boolean initial, MNIST0_InternalData internal) throws IOException {
		if (initial)
		{

			
			for(int i=0; i<128; i++)
			{
				for(int I=0; I<576; I++)
				{
					initWeights0[I][i] = internal.weights6[I][i];
					initWeights1[I][i] = internal.weights6[I][i];
					initWeights2[I][i] = internal.weights6[I][i];
					initWeights3[I][i] = internal.weights6[I][i];
					initWeights4[I][i] = internal.weights6[I][i];
					initWeights5[I][i] = internal.weights6[I][i];
					initWeights6[I][i] = internal.weights6[I][i];
					initWeights7[I][i] = internal.weights6[I][i];
					initWeights8[I][i] = internal.weights6[I][i];
					initWeights9[I][i] = internal.weights6[I][i];
					initWeights10[I][i] = internal.weights6[I][i];
					average_weights6[I][i] = 0.0;
				}
			}
			
		   
			loaddeltas(0);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights0[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			loaddeltas(1);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights1[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			loaddeltas(2);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights2[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			loaddeltas(3);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights3[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			loaddeltas(4);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights4[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			loaddeltas(5);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights5[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			
			loaddeltas(6);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights6[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			
			loaddeltas(7);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights7[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			
			loaddeltas(8);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights8[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			loaddeltas(9);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights9[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			
			loaddeltas(10);
			for(int i=0;i<num1.size();i++)
			{
							
				initWeights10[num1.get(i)][num0.get(i)]= internal.weights6[num1.get(i)][num0.get(i)]+(num2.get(i));
			}
			
			for(int i=0; i<128; i++)
			{
				for(int I=0; I<576; I++)
				{
					
					average_weights6[I][i] = (internal.weights6[I][i] + initWeights0[I][i] + initWeights1[I][i] + initWeights2[I][i] + initWeights3[I][i] + initWeights4[I][i] + 	initWeights5[I][i] + initWeights6[I][i] + initWeights7[I][i] + + initWeights8[I][i] + + initWeights9[I][i]);
					average_weights6[I][i] = (average_weights6[I][i]/11.0);
				}
			}
			
			EXPERTS = new ArrayList<Integer>();
			EXPERTS.add(6);
			EXPERTS.add(8);
			EXPERTS.add(9);
			
		}
		
		
	}

	int run(double[][][] input, int lab) throws IOException
	{

		init(false, this.internal);
		//  layer 0: conv2d_1
		double[][][] layer0=new double[26][26][2];
		for(int i=0; i<26; i++)
			for(int j=0; j<26; j++)
				for(int k=0; k<2; k++)
				{
					layer0[i][j][k]=internal.biases0[k];
					for(int I=0; I<3; I++)
						for(int J=0; J<3; J++)
							for(int K=0; K<1; K++)
								layer0[i][j][k]+=internal.weights0[I][J][K][k]*input[i+I][j+J][K];
				}

		//  layer 1: activation_1
		double[][][] layer1=new double[26][26][2];
		for(int i=0; i<26; i++)
			for(int j=0; j<26; j++)
				for(int k=0; k<2; k++)
					if(layer0[i][j][k]>0) 
					{
						layer1[i][j][k]=layer0[i][j][k];



					}
					else 
					{
						layer1[i][j][k]=0;


					}


		//  layer 2: conv2d_2
		double[][][] layer2=new double[24][24][4];
		for(int i=0; i<24; i++)
			for(int j=0; j<24; j++)
				for(int k=0; k<4; k++)
				{
					layer2[i][j][k]=internal.biases2[k];
					for(int I=0; I<3; I++)
						for(int J=0; J<3; J++)
							for(int K=0; K<2; K++)
								layer2[i][j][k]+=internal.weights2[I][J][K][k]*layer1[i+I][j+J][K];
				}

		//  layer 3: activation_2
		double[][][] layer3=new double[24][24][4];
		for(int i=0; i<24; i++)
			for(int j=0; j<24; j++)
				for(int k=0; k<4; k++)
					if(layer2[i][j][k]>0)
					{
						layer3[i][j][k]=layer2[i][j][k];

					}
					else 
					{
						layer3[i][j][k]=0;

					}

		//  layer 4: max_pooling2d_1
		double[][][] layer4=new double[12][12][4];
		for(int i=0; i<12; i++)
			for(int j=0; j<12; j++)
				for(int k=0; k<4; k++)
				{
					layer4[i][j][k]=0;
					for(int I=i*2; I<(i+1)*2; I++)
						for(int J=j*2; J<(j+1)*2; J++)
							if(layer3[I][J][k]>layer4[i][j][k]) layer4[i][j][k]=layer3[I][J][k];
				}

		//  layer 5: flatten_1
		double[] layer5=new double[576];
		for(int i=0; i<576; i++)
		{
			int d0=i/48;
			int d1=(i%48)/4;
			int d2=i-d0*48-d1*4;
			layer5[i]=layer4[d0][d1][d2];
		}

		double[] layer6=new double[128];
		
		for(int i=0; i<128; i++)
		{
			layer6[i]=internal.biases6[i];
			for(int I=0; I<576; I++)
			{
				if (lab > 0 && lab < 10)
					train_prec = train_precision[lab];
				
				
				
				if (lab == 0)
				{
					layer6[i]+=initWeights0[I][i]*layer5[I];
				//	train_prec = 0.971;
					
					
				}
				if (lab == 1)
				{
					layer6[i]+=initWeights1[I][i]*layer5[I];
				//	train_prec = 0.9849;
				}
				if (lab == 2)
				{
					layer6[i]+=initWeights2[I][i]*layer5[I];
				//	train_prec = 0.6278;
				}
				if (lab == 3)
				{
					layer6[i]+=initWeights3[I][i]*layer5[I];
				//	train_prec = 0.9502;
				}
				if (lab == 4)
				{
					layer6[i]+=initWeights4[I][i]*layer5[I];
				//	train_prec = 0.677;
				}
				if (lab == 5)
				{
					layer6[i]+=initWeights5[I][i]*layer5[I];
				//	train_prec = 0.9135;
				}
				if (lab == 6)
				{
					layer6[i]+=initWeights6[I][i]*layer5[I];
				//	train_prec = 0.99;
				}
				if (lab == 7)
				{
					layer6[i]+=initWeights7[I][i]*layer5[I];
				//	train_prec = 0.9571;
				}
				if (lab == 8)
				{
					layer6[i]+=initWeights8[I][i]*layer5[I];
				//	train_prec = 0.9748;
				}
				if (lab == 9)
				{
					layer6[i]+=initWeights9[I][i]*layer5[I];
				//	train_prec = 0.9312;
				}
				
				if (lab == 10)
				{
					layer6[i]+=initWeights10[I][i]*layer5[I];
					
				}
				
				if (lab == -1)
					layer6[i]+=internal.weights6[I][i]*layer5[I];
				
				if (lab == 100)
				{
					layer6[i]+=average_weights6[I][i]*layer5[I];
				}
				
				


			}
		}
		
		//  layer 7: activation_3
		double[] layer7=new double[128];

		for(int i=0; i<128; i++)
		{

			if(layer6[i]>0) 
			{
				layer7[i]=layer6[i];

			}
			else {
				layer7[i]=0;

			}
		}
         
	
		double[] layer8=new double[10];
		for(int i=0; i<10; i++)
		{
			layer8[i]=internal.biases8[i];
			for(int I=0; I<128; I++)
				layer8[i]+=internal.weights8[I][i]*layer7[I];
		}

		//  layer 9: activation_4
		int ret=0;
		double res=-100000;
		for(int i=0; i<10;i++)
		{
			if(layer8[i]>res)
			{
				res=layer8[i];
				ret=i;
			}
		}
        label_conf = res;
		return ret;
	}



	public static void main(String[] args) throws IOException{
		
		try {

			if (MODEL == 1)
				path = "C:\\Users\\dgopinat\\eclipse-workspace\\mnist_example\\mnist_example\\data\\poisoned\\";
			if (MODEL == 0)
				path = "C:\\Users\\dgopinat\\eclipse-workspace\\mnist_example\\mnist_example\\data\\low-quality\\";
			if (MODEL == 2)
				path = "C:\\Users\\dgopinat\\eclipse-workspace\\mnist_example\\mnist_example\\data\\high-quality\\";
			

			System.out.println("PATH:" + path);

			MNIST0_InternalData data = new MNIST0_InternalData(path,"weights0.txt","weights2.txt","weights6.txt","weights8.txt","biases0.txt","biases2.txt","biases6.txt","biases8.txt");
			
			MNIST0_DNNt_pattern_based model = new MNIST0_DNNt_pattern_based(data);

			String labelFile = path + "mnist_test_labels.txt";
			
			File file = new File(labelFile); 
			BufferedReader br = new BufferedReader(new FileReader(file)); 
			String st; 
			Integer[] labels = new Integer[60000];
			int index = 0;
			while ((st = br.readLine()) != null) {
				labels[index] = Integer.valueOf(st);
				index++;
			}

			br.close();
			
			String inputFile = path + "mnist_test.txt";
			
			file = new File(inputFile); 
			br = new BufferedReader(new FileReader(file)); 
			int count = 0;
			int pass = 0;
			int fail = 0;

			while ((st = br.readLine()) != null) {
				//  System.out.println("INPUT:" + st); 
				String[] values = st.split(",");
				double[][][] input = new double[28][28][1];
				index = 0;
				while (index < values.length) {
					for (int i = 0; i < 28 ; i++)
						for (int j = 0; j < 28; j++)
							for (int k = 0; k < 1; k++)
							{
								Double val = Double.valueOf(values[index]);
								index++;
								//  input[i][j][k] = (double)(val/255.0);
								input[i][j][k] = (double)(val);
							}
				}

				boolean run = true;
				
				if (labels[count] != LABEL) //Targetted repair
					run = false;
				
				if (LABEL == -1)
					run = true;
				//run = true;
				
				if (run)
				{
					int label = -1;
					int origlabel = -1; 
					
					//ORIGINAL LABEL
					origlabel = model.run(input,-1); // original
					
					if (COMB_METHOD == "AVERAGE")
						label = model.run(input,100); // AVERAGE
					if (COMB_METHOD == "FULL")
						label = model.run(input,10); // FULL
					

					ArrayList<Integer> experts = new ArrayList<Integer>();
					ArrayList<Double> confidence = new ArrayList<Double>();
					ArrayList<Double> tr_precs = new ArrayList<Double>();
					Integer[] Votes = new Integer[10];
					for (int i = 0 ; i < 10; i++)
					{
							Votes[i]=0;
					}
						
					if (SELECT_EXPERTS == "YES")
					{
						for (int i = 0; i < exp_labs.length;i++)
							EXPERTS.add(exp_labs[i]);
					}
					else
					{
						for (int i = 0; i < 10;i++)
							EXPERTS.add(i);
					}
						
						
					for (int j = 0 ; j < EXPERTS.size(); j++)
					{
							int i = EXPERTS.get(j);
							int lab = model.run(input,i);
							if (lab == i)
							{
								experts.add(i);
								confidence.add(label_conf);
								tr_precs.add(train_prec);
							}
							Votes[lab]++;
					}
					 	
					Votes[origlabel]++;
					
					if (COMB_METHOD == "NAIVE")
					{
						if(experts.size() == 1)
						{
								label = experts.get(0);
						}
						else
								label = origlabel;
					}

					// COMBINED SCORE: PRECISION, VOTES, CONFIDENCE
					if ((COMB_METHOD == "PREC") || (COMB_METHOD == "VOTES") || (COMB_METHOD == "CONF") || (COMB_METHOD == "P+V+C"))
					{
						if(experts.size() == 1)
						{
							label = experts.get(0);
						}
						
						if(experts.size() == 0)
						{
							label = origlabel;
						}
	                    	
						if (experts.size() > 1)
						{
							double maxPrec = 0.0;
							int maxVotes = 0;
							double maxConf = 0.0;
						
							ArrayList<Double> score = new ArrayList<Double>();
							for (int k=0; k < experts.size(); k++)
								score.add(0.0);
							
							int max = -1;
						    if ((COMB_METHOD == "PREC") || (COMB_METHOD == "P+V+C"))
						    {
						    	for (int i=0; i < experts.size() ; i++) 
						    	{
						    		Integer e = experts.get(i);
							
						    		if (tr_precs.get(i)  > maxPrec)
						    		{
									
						    			maxPrec = tr_precs.get(i) ;
						    			max = i;
						    		}
								
								
						    	}
						    	score.set(max, score.get(max)+1.0);
						    }
						    
						    if ((COMB_METHOD == "VOTES") || (COMB_METHOD == "P+V+C"))
						    {
						    	max = -1;
							
						    	for (int i=0; i < experts.size() ; i++) 
						    	{
						    		Integer e = experts.get(i);
							
						    		if ( Votes[e]  > maxVotes)
						    		{
									
						    			maxVotes = Votes[e] ;
						    			max = i;
						    		}
								
								
						    	}
						    	score.set(max, score.get(max)+1.0);
                              							
						    }
						    
						    if ((COMB_METHOD == "CONF") || (COMB_METHOD == "P+V+C"))
						    {
						    	max = -1;
						    	for (int i=0; i < experts.size() ; i++) 
						    	{
						    		Integer e = experts.get(i);
							
						    		if  (confidence.get(i)  > maxConf)
						    		{
						    			maxConf = confidence.get(i);
						    			max = i;
						    		}
								
								
						    	}
						    	score.set(max, score.get(max)+1);
							
						    }
							double maxScore = -1.0;
							for (int i=0; i < experts.size() ; i++)
							{
								Integer e = experts.get(i);
								System.out.println("Expert:"+ e +",VOTES:" + Votes[e] + ",CONF:" + confidence.get(i) + ",PREC:" + tr_precs.get(i) + ",SCORE:" + score.get(i));	
								
								if (score.get(i)  > maxScore)
								{
									maxScore = score.get(i);
									label = e;
								}
								
								
							}
							
							
							
					  }
					
					    
					if (label == -1)
						label = origlabel;
						
					}
					
					if (label == labels[count])
					{

						pass++;
						System.out.println("PASS:," + "IDEAL:"+ labels[count] + ", ORIG:" + origlabel + ",MODEL:" + label);
						

					}
					else
					{
						fail++;
						System.out.println("FAIL:," + "IDEAL:"+ labels[count] + ", ORIG:" + origlabel + ",MODEL:" + label);
							


					}
				}
				
				count++;

			}
			double accuracy = (((double)pass)/(pass + fail))*100.0;
			System.out.println("OVERAL MODEL ACCURACY:" + accuracy);

			System.out.println("PASS:"+ pass + ", FAIL:"+ fail + ", accuracy:"+ accuracy);

		
			br.close();
			
		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class Program
    {
        static void Main(string[] args)
        {
            
            //layers
            int[] layers = new[] { 2, 6, 6, 1 };
            var nn = new NeuralNetwork(layers)
            {
                Epocs = 10000,              
                Alpha = 0.8,
                Beta = 0.2,
                MomentumParameter = true,
                Rnd = new Random(12345)         
            };

           
            //Data XOR
            var training = new double[][]
            {
                new double[]{ 1, 0, 1 },
                new double[]{ 0, 1, 1 },
                new double[]{ 1, 1, 0 },
                new double[]{ 0, 0, 0 },
                new double[]{ 1, 1, 0 },
                new double[]{ 0, 0, 0 },
                new double[]{ 1, 0, 1 },
                new double[]{ 0, 1, 1 },
            };


            //Take the first 2 columns as input, and last 1 column as target y (the expected label)
            var input = new double[training.GetLength(0)][];
            for (int i = 0; i < training.GetLength(0); i++)
            {
                input[i] = new double[layers[0]];
                for (int j = 0; j < layers[0]; j++)
                    input[i][j] = training[i][j];
            }
                
            //Create the expected label array
            var y = new double[training.GetLength(0)];
            for (int i = 0; i < training.GetLength(0); i++)
                y[i] = training[i][layers[0]];


            //Begin training the network to learn the function that matches our data.
            nn.Train(input, y);

            //Confirm Neural Network its worked
            Console.WriteLine($"The network learned XOR(0,0)={nn.Predict(new[] { 0.0, 0.0})[0]}");
            Console.WriteLine($"The network learned XOR(0,1)={nn.Predict(new[] { 0.0, 1.0})[0]}");
            Console.WriteLine($"The network learned XOR(1,0)={nn.Predict(new[] { 1.0, 0.0 })[0]}");
            Console.WriteLine($"The network learned XOR(1,1)={nn.Predict(new[] { 1.0, 1.0 })[0]}");


            nn.BeeTraining(input, y);
            // Confirm ABC its worked
            Console.WriteLine($"The network learned with ABC XOR(0,0)={nn.Predict(new[] { 0.0, 0.0 })[0]}");
            Console.WriteLine($"The network learned with ABC XOR(0,1)={nn.Predict(new[] { 0.0, 1.0 })[0]}");
            Console.WriteLine($"The network learned with ABC XOR(1,0)={nn.Predict(new[] { 1.0, 0.0 })[0]}");
            Console.WriteLine($"The network learned with ABC XOR(1,1)={nn.Predict(new[] { 1.0, 1.0 })[0]}");

            Console.WriteLine("press any key to continue");
            Console.ReadKey(true);
        }
    }
}

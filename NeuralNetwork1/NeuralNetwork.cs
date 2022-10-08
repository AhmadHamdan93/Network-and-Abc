﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    
    public class NeuralNetwork
    {
  
        private Neuron[][] Neurons;
        private double[][,] Weights;
        private double[][,] momentum;
        //--------------------------------------
        private double[] errors; // for save MSE error in every loop ,its size equal Epocs 
        private double[] errors_RMSE; // for save RMSE error in eveery loop , its size equal Epocs

        private double[,] populations;
        public int Rows { get; set; } = 50;


        public int Epocs { get; set; } = 10000;
        public bool MomentumParameter { get; set; } = true;
        public double Beta { get; set; } = 0.9;

        /// Controls the learning rate.  Increasing for larger jumps but too high may prevent convergence.
        public double Alpha { get; set; } = 0.9;
        // -------------------------------------

        private int LastLayer;
        public Random Rnd { get; set; } = new Random();


        public NeuralNetwork(int[] layers)
        {
            LastLayer = layers.Length - 1;

            //All nodes in the network will be neurons
            Neurons = new Neuron[layers.Length][];
            for (int l = 0; l < layers.Length; l++)
            {
                Neurons[l] = new Neuron[layers[l]];
                //Initialize each layers nodes
                for (int n = 0; n < layers[l]; n++)
                    Neurons[l][n] = new Neuron();
            }

            // Initialize error array
            errors = new double[Epocs];
            errors_RMSE = new double[Epocs];
        }

        private void InitializeWeights()
        {
            int layers = Neurons.GetLength(0);
            Weights = new double[layers - 1][,];
            momentum = new  double[layers - 1][,];
            for (int l = 0; l < layers - 1; l++)
            {
                Weights[l] = new double[Neurons[l].Length, Neurons[l + 1].Length];  //2x3, 3x1
                momentum[l] = new double[Neurons[l].Length, Neurons[l + 1].Length];
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    for (int j = 0; j < Neurons[l + 1].Length; j++)
                    {
                        Weights[l][n, j] = WeightFunction(l + 1);
                        momentum[l][n, j] = 0;
                    }
                }
            }
        }

        private double WeightFunction(int layer)
        {
            //Randomly sample a uniform distribution in the range [-b,b] where b is:
            //where fanIn is the number of input units in the weights and
            //fanOut is the number of output units in thes weights
            var fanIn = (layer > 0) ? Neurons[layer - 1].Length : 0;
            var fanOut = Neurons[layer].Length;
            var b = Math.Sqrt(6) / Math.Sqrt(fanIn + fanOut);
            return Rnd.NextDouble() * 2 * b - b;

        }

        public void Train(double[][] input, double [] y)
        {
            InitializeWeights();
            int epoc = Epocs;
            var cost = new double[Neurons[LastLayer].Length];
            
            //------------------
            int columns = getLen(Weights) + 1;
            populations = new double[Rows, columns];
            //------------------

            while (epoc-- > 0)
            {
                //Loop through each input, and compute the prediction
                for (int i = 0; i < input.GetLength(0); i++)
                {

                    //FeedForward
                    var output = Predict(input[i]);


                    //Compute the error in the output layer
                    for (int n = 0; n < Neurons[LastLayer].Length; n++)
                    {
                        //Compute the error at the output layer
                        cost[n] = Neurons[LastLayer][n].output - y[i];

                        //Assign the error to the output layer's error
                        //But becuase the cost function is entopy then delta = error
                        Neurons[LastLayer][n].error = cost[n];

                        // save error for MAE :
                        errors[epoc] += Math.Abs(cost[n]);  //cost[n] * cost[n];
                        // save error for RMSE :
                        errors_RMSE[epoc] += cost[n] * cost[n];
                    }

                    //Backpropagate the error through the network
                    BackPropagate();

                    //SGD Method
                    //Adjust the weights by the amount of this error
                    for (int l = 0; l <= LastLayer - 1; l++)
                    {
                        for (int j = 0; j < Neurons[l].Length; j++)
                            for (int k = 0; k < Neurons[l + 1].Length; k++)
                            {
                                if (MomentumParameter)
                                {
                                    momentum[l][j, k] = Beta * momentum[l][j, k] + (Alpha * Neurons[l][j].output * Neurons[l + 1][k].error);
                                    Weights[l][j,k] -= momentum[l][j, k];
                                }
                                else
                                    Weights[l][j, k] -= (Alpha * Neurons[l][j].output * Neurons[l + 1][k].error);
                            }
                    }
                }

                MAE_Error(input.GetLength(0), epoc);
                RMSE_Error(input.GetLength(0), epoc);
                
                // must be here save weights as vector
                if (epoc < Rows)
                {
                    double[] vector = convert3D_to_1D(Weights);
                    BuildPopulation(vector, errors_RMSE[epoc], epoc);    //  errors[epoc]
                }
            }
            Console.WriteLine("'MAE' The Mean Absolute Error for Neural Network is : " + errors[0]);
            Console.WriteLine("'RMSE' Root Mean Square Error for Neural Network is : " + errors_RMSE[0]);
            Console.WriteLine();
            //MSE_Error(input.GetLength(0), epoc);
        }

        public void BuildPopulation(double[] v, double error, int idx)
        {
            int len = v.GetLength(0) - 1;
            for (int i = 0; i < len; i++)
            {
                populations[idx,i] = v[i];
            }
            populations[idx, len] = error;
        }

        public void MAE_Error(int n, int idx)
        {
            errors[idx] = errors[idx] / n;
        }

        public void RMSE_Error(int n, int idx)
        {
            errors_RMSE[idx] = errors_RMSE[idx] / n;
            errors_RMSE[idx] = Math.Sqrt(errors_RMSE[idx]);
        }

        public double[] Predict(double[] input)
        {
            //Forward propagation
            //Fill the first layers output neurons with input data
            for (int d = 0; d < Neurons[0].Length; d++)
                Neurons[0][d].output = input[d];

            //Feed forward phase
            for (int l = 1; l < Neurons.GetLength(0); l++)
            {
                //Now compute layer l, n is each neuron in layer l
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    //Compute neuron n in layer l
                    double sum = 0;

                    //Iterate over previous layers outputs and weights
                    //j is each of the previous layers neuron
                    for (int j = 0; j < Neurons[l - 1].Length; j++)
                        sum += (Neurons[l - 1][j].output * Weights[l - 1][j, n]);

                    //Store the weighted inputs on input.
                    Neurons[l][n].input = sum;

                    //The output is the sigmoid of the weighted input 
                    Neurons[l][n].output = Sigmoid(Neurons[l][n].input +1);
                }
            }

            //prepare a vector of outputs to return
            var outputlayer = Neurons.GetLength(0) - 1;
            var output = new double[Neurons[outputlayer].Length];
            for (int n = 0; n < output.Length; n++)
                output[n] = Neurons[outputlayer][n].output;

            return output;
        }


        public void BackPropagate()
        {
            //From right to left (output to input)
            for (int l = LastLayer - 1; l > 0; l--)
            {
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    //matrix form
                    //error in this layer = dot product( weights in next layer,  error in next layer) * sigmaprime(weighted inputs of this layer)

                    //Sum the product of the weight * error in L+1
                    double sum = 0.0;
                    for (int m = 0; m < Neurons[l + 1].Length; m++)
                    {
                        //Weights of L is actually the L + 1 layer
                        sum += (Weights[l][n, m] * Neurons[l + 1][m].error);
                    }

                    Neurons[l][n].error = sum * SigmoidDerivate(Neurons[l][n].input + 1); 
                   
                }
            }
        }

        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }


        private static double SigmoidDerivate(double x)
        {
            return Sigmoid(x) * (1.0 - Sigmoid(x));
        }

        public int getLen(double[][,] a)
        {
            int length = 0;
            int x, y, z;
            x = a.GetLength(0);
            for (int i = 0; i < x; i++)
            {
                y = a[i].GetLength(0);
                z = a[i].GetLength(1);
                length += y * z;
            }
            return length;
        }

        public double[] convert3D_to_1D(double[][,] a)
        {
            int D = getLen(a);
            double[] row = new double[D + 1];
            int t = 0;
            int l = a.GetLength(0);
            for (int i = 0; i < l; i++)
                for (int j = 0; j < a[i].GetLength(0); j++)
                    for (int k = 0; k < a[i].GetLength(1); k++)
                    {
                        row[t] = a[i][j, k];
                        t++;
                    }

            return row;
        }

        public double[][,] convert1D_to_3D(double[] a)
        {
            // here we must already know size 3D array in every level
            // for reason we based on 'w' array for know it
            double[][,] array = new double[Weights.GetLength(0)][,];
            for (int i = 0; i < Weights.GetLength(0); i++)
                array[i] = new double[Weights[i].GetLength(0), Weights[i].GetLength(1)];

            // ----------------------------
            int t = 0;
            for (int i = 0; i < Weights.GetLength(0); i++)
                for (int j = 0; j < Weights[i].GetLength(0); j++)
                    for (int k = 0; k < Weights[i].GetLength(1); k++)
                    {
                        array[i][j, k] = a[t];
                        t++;
                    }
            return array;
        }

        public void BeeTraining(double[][] input, double[] y)
        {
            // search by bee colony
            // population , input , y 
            Bee bee = new Bee(populations, input, y, Weights, Neurons);
            bee.Search();
            // // save data on weights matrix
            double[] arr = bee.getBestSolution();
            Weights = convert1D_to_3D(arr);

        }

    }
}

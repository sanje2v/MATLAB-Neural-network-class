classdef NeuralNet < handle
    % Written by: Sanjeev Sharma
    
    properties (Access='private')
        % Activation function and its partial derivative
        f2 = @(x) (1-exp(-x))./(1+exp(-x));
        f2_dash = @(x) 0.5 * (1 - x.^2);
        
        m_Learning_Rate;
        m_Weight_HiddenLayer;
        m_Weight_OutputLayer;
        
        m_CycleError = 0;
    end
    
    methods (Static)
        function [ output_weights ] = GenerateRandomWeights(rows, columns)
            output_weights = 0.5 - rand(rows, columns);
        end
    end
    
    methods  (Access='public')     
        function obj = NeuralNet(learning_rate, ...
                                    num_input_layer, ...
                                    num_hidden_layer, ...
                                    num_output_layer)
            obj.m_Learning_Rate = learning_rate;
            obj.m_Weight_HiddenLayer = NeuralNet.GenerateRandomWeights(num_hidden_layer, num_input_layer + 1);
            obj.m_Weight_OutputLayer = NeuralNet.GenerateRandomWeights(num_output_layer, num_hidden_layer + 1);
        end
        
        function StartNewCycle(obj)
            obj.m_CycleError = 0;
        end
        
        function cycle_error = EndCycle(obj)
            cycle_error = obj.m_CycleError;
        end
        
        function Train(obj, input, expected_output)
            Input_Data = vertcat(input, -1);
            if size(expected_output, 1) == 1
                expected_output = expected_output';
            end
            
            % Feed forward
            Output_HiddenLayer = vertcat(obj.f2(obj.m_Weight_HiddenLayer * Input_Data), -1);
            Output_OutputLayer = obj.f2(obj.m_Weight_OutputLayer * Output_HiddenLayer);

            % Calculate error for output layer
            PatternError_Output = expected_output - Output_OutputLayer;
            Error_OutputLayer = obj.f2_dash(Output_OutputLayer) .* PatternError_Output;

            % Calculate error for hidden layer
            SumProduct = zeros(size(obj.m_Weight_HiddenLayer, 1), 1);
            for i = 1:numel(SumProduct)
                SumProduct(i) = sum(Error_OutputLayer .* obj.m_Weight_OutputLayer(:, i));
            end
            Error_HiddenLayer = obj.f2_dash(Output_HiddenLayer(1:end - 1)) .* SumProduct;

            % Update output and hidden layer weights
            obj.m_Weight_OutputLayer = obj.m_Weight_OutputLayer + ...
                                        obj.m_Learning_Rate * Error_OutputLayer * Output_HiddenLayer';
            obj.m_Weight_HiddenLayer = obj.m_Weight_HiddenLayer + ...
                                        obj.m_Learning_Rate * Error_HiddenLayer * Input_Data';

            obj.m_CycleError = obj.m_CycleError + sum(0.5 * (PatternError_Output .^ 2));
        end
        
        function output_values = GetOutput(obj, input)
            Input_Data = vertcat(input, -1);
            
            Output_HiddenLayer = vertcat(obj.f2(obj.m_Weight_HiddenLayer * Input_Data), -1);
            output_values = obj.f2(obj.m_Weight_OutputLayer * Output_HiddenLayer);
        end
    end
end


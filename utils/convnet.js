// Utility functions
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
  
  function dsigmoid(y) {
    return y * (1 - y);
  }
  
  function randomWeight() {
    return Math.random() * 2 - 1;
  }
  
  function relu(x) {
    return Math.max(0, x);
  }
  
  function drelu(x) {
    return x > 0 ? 1 : 0;
  }
  
  function softmax(scores) {
    const expScores = scores.map(score => Math.exp(score));
    const sumExpScores = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(expScore => expScore / sumExpScores);
  }
  
  // Helper function to perform convolution with residual connections
  function convolveWithResidual(input, kernel, stride = 1, padding = 0) {
    const outputSize = Math.floor((input.length - kernel.length + 2 * padding) / stride) + 1;
    const output = new Array(outputSize).fill(0).map(() => new Array(outputSize).fill(0));
  
    for (let i = 0; i < outputSize; i++) {
        for (let j = 0; j < outputSize; j++) {
            let sum = 0;
            for (let ki = 0; ki < kernel.length; ki++) {
                for (let kj = 0; kj < kernel[ki].length; kj++) {
                    const ii = i * stride + ki - padding;
                    const jj = j * stride + kj - padding;
                    if (ii >= 0 && ii < input.length && jj >= 0 && jj < input.length) {
                        sum += input[ii][jj] * kernel[ki][kj];
                    }
                }
            }
            output[i][j] = relu(sum + input[i * stride][j * stride]);  // Adding residual connection
        }
    }
    return output;
  }
  
  // Helper function for max pooling
  function maxPool(input, poolSize = 2, stride = 2) {
    const outputSize = Math.floor((input.length - poolSize) / stride) + 1;
    const output = new Array(outputSize).fill(0).map(() => new Array(outputSize).fill(0));
  
    for (let i = 0; i < outputSize; i++) {
        for (let j = 0; j < outputSize; j++) {
            let maxVal = -Infinity;
            for (let pi = 0; pi < poolSize; pi++) {
                for (let pj = 0; pj < poolSize; pj++) {
                    const ii = i * stride + pi;
                    const jj = j * stride + pj;
                    maxVal = Math.max(maxVal, input[ii][jj]);
                }
            }
            output[i][j] = maxVal;
        }
    }
    return output;
  }
  
  class ConvNet {
    constructor(input_dim, kernel_dim, num_kernels, hidden_nodes, output_nodes) {
        this.input_dim = input_dim; // Assuming square input, e.g., 28x28
        this.kernel_dim = kernel_dim; // Square kernel, e.g., 3x3
        this.num_kernels = num_kernels; // Number of kernels/filters
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;
  
        this.kernels = new Array(this.num_kernels).fill(0).map(() => 
            new Array(this.kernel_dim).fill(0).map(() => 
                new Array(this.kernel_dim).fill(0).map(randomWeight)));
  
        const conv_output_dim = this.input_dim - this.kernel_dim + 1;
        const pooled_dim = Math.floor(conv_output_dim / 2); // After 2x2 pooling
  
        this.weights_fc = new Array(this.hidden_nodes).fill(0).map(() => 
            new Array(this.num_kernels * pooled_dim * pooled_dim).fill(0).map(randomWeight));
        this.weights_ho = new Array(this.output_nodes).fill(0).map(() => 
            new Array(this.hidden_nodes).fill(0).map(randomWeight));
  
        this.bias_fc = new Array(this.hidden_nodes).fill(0).map(randomWeight);
        this.bias_o = new Array(this.output_nodes).fill(0).map(randomWeight);
  
        // Parameters for optimization
        this.learning_rate = Data.training_stats.learning_rate; // Typically lower for Adam
        this.gradient_clip = Data.training_stats.gradient_clip; // Gradient clipping threshold
  
        // Adam optimizer parameters
        this.beta1 = Data.training_stats.beta1;
        this.beta2 = Data.training_stats.beta2;
        this.epsilon = Data.training_stats.epsilon;
        this.t = 0; // time step
  
        // Initialize Adam moment estimates
        this.m_weights_fc = this.weights_fc.map(row => row.map(() => 0));
        this.v_weights_fc = this.weights_fc.map(row => row.map(() => 0));
        this.m_weights_ho = this.weights_ho.map(row => row.map(() => 0));
        this.v_weights_ho = this.weights_ho.map(row => row.map(() => 0));
        this.m_bias_fc = this.bias_fc.map(() => 0);
        this.v_bias_fc = this.bias_fc.map(() => 0);
        this.m_bias_o = this.bias_o.map(() => 0);
        this.v_bias_o = this.bias_o.map(() => 0);
    }
  
    predict(input_array) {
        // Convert the input array into a 2D array
        const input2D = [];
        for (let i = 0; i < this.input_dim; i++) {
            input2D.push(input_array.slice(i * this.input_dim, (i + 1) * this.input_dim));
        }
  
        // Convolutional layer with residual connections
        const convOutputs = this.kernels.map(kernel => {
            const convOutput = convolveWithResidual(input2D, kernel);
            return convOutput.map(row => row.map(relu));
        });
  
        // Pooling layer
        const pooledOutputs = convOutputs.map(convOutput => maxPool(convOutput));
  
        // Flatten pooled output
        const flattened = pooledOutputs.flat(2);
  
        // Fully connected layer
        let hidden = this.weights_fc.map((weights_row, i) => 
            sigmoid(weights_row.reduce((sum, weight, j) => sum + weight * flattened[j], this.bias_fc[i])));
  
        // Output layer
        let scores = this.weights_ho.map((weights_row, i) => 
            weights_row.reduce((sum, weight, j) => sum + weight * hidden[j], this.bias_o[i]));
  
        // Apply softmax to get probabilities
        let probabilities = softmax(scores);
  
        return probabilities;
    }
    getActivations(input_array) {
        const input2D = [];
        for (let i = 0; i < this.input_dim; i++) {
          input2D.push(input_array.slice(i * this.input_dim, (i + 1) * this.input_dim));
        }
    
        const convOutputs = this.kernels.map(kernel => {
          const convOutput = convolveWithResidual(input2D, kernel);
          return convOutput.map(row => row.map(relu));
        });
    
        const pooledOutputs = convOutputs.map(convOutput => maxPool(convOutput));
        const flattened = pooledOutputs.flat(2);
    
        const hidden = this.weights_fc.map((weights_row, i) => 
          sigmoid(weights_row.reduce((sum, weight, j) => sum + weight * flattened[j], this.bias_fc[i]))
        );
    
        const scores = this.weights_ho.map((weights_row, i) => 
          weights_row.reduce((sum, weight, j) => sum + weight * hidden[j], this.bias_o[i])
        );
    
        const probabilities = softmax(scores);
    
        return {
          input: input2D.flat(),
          convOutputs: convOutputs.flat(2),
          pooledOutputs: pooledOutputs.flat(2),
          flattened,
          hidden,
          output: probabilities
        };
      }
    
      getWeightsAndBiases() {
        return {
          inputToHidden: this.weights_fc,
          hiddenToOutput: this.weights_ho,
          biases_fc: this.bias_fc,
          biases_o: this.bias_o
        };
      }

      getFeatureMaps(input_array) {
        // Convert the input array into a 2D array
        const input2D = [];
        for (let i = 0; i < this.input_dim; i++) {
            input2D.push(input_array.slice(i * this.input_dim, (i + 1) * this.input_dim));
        }

        // Convolutional layer with residual connections
        const convOutputs = this.kernels.map(kernel => {
            const convOutput = convolveWithResidual(input2D, kernel);
            return convOutput.map(row => row.map(relu));
        });

        // Get the dimensions of the output feature maps
        const convOutputSize = convOutputs[0].length; // Assuming all kernels produce the same output size

        // Return the raw feature maps in a 2D format
        return {
            featureMaps: convOutputs,
            dimensions: {
                width: convOutputSize,
                height: convOutputSize,
                numKernels: this.num_kernels
            }
        };
    }
  
    train(input_array, target_array) {
        // Convert the input array into a 2D array
        const input2D = [];
        for (let i = 0; i < this.input_dim; i++) {
            input2D.push(input_array.slice(i * this.input_dim, (i + 1) * this.input_dim));
        }
  
        // Convolutional layer with residual connections
        const convOutputs = this.kernels.map(kernel => {
            const convOutput = convolveWithResidual(input2D, kernel);
            return convOutput.map(row => row.map(relu));
        });
  
        // Pooling layer
        const pooledOutputs = convOutputs.map(convOutput => maxPool(convOutput));
  
        // Flatten pooled output
        const flattened = pooledOutputs.flat(2);
  
        // Fully connected layer
        let hidden = this.weights_fc.map((weights_row, i) => 
            sigmoid(weights_row.reduce((sum, weight, j) => sum + weight * flattened[j], this.bias_fc[i])));
  
        // Output layer
        let outputs = this.weights_ho.map((weights_row, i) => 
            sigmoid(weights_row.reduce((sum, weight, j) => sum + weight * hidden[j], this.bias_o[i])));
  
        // Increment time step
        this.t += 1;
  
        // Calculate output errors
        let output_errors = outputs.map((output, i) => target_array[i] - output);
  
        // Calculate gradients for the output layer
        let gradients = outputs.map((output, i) => dsigmoid(output) * output_errors[i]);
  
        // Gradient clipping
        gradients = gradients.map(grad => Math.max(-this.gradient_clip, Math.min(this.gradient_clip, grad)));
  
        // Update weights and biases for the output layer using Adam
        this.weights_ho.forEach((weights_row, i) => {
            weights_row.forEach((weight, j) => {
                const grad = gradients[i] * hidden[j];
                this.m_weights_ho[i][j] = this.beta1 * this.m_weights_ho[i][j] + (1 - this.beta1) * grad;
                this.v_weights_ho[i][j] = this.beta2 * this.v_weights_ho[i][j] + (1 - this.beta2) * grad * grad;
                const m_hat = this.m_weights_ho[i][j] / (1 - Math.pow(this.beta1, this.t));
                const v_hat = this.v_weights_ho[i][j] / (1 - Math.pow(this.beta2, this.t));
                this.weights_ho[i][j] += this.learning_rate * m_hat / (Math.sqrt(v_hat) + this.epsilon);
            });
        });
  
        this.bias_o = this.bias_o.map((bias, i) => {
            const grad = gradients[i];
            this.m_bias_o[i] = this.beta1 * this.m_bias_o[i] + (1 - this.beta1) * grad;
            this.v_bias_o[i] = this.beta2 * this.v_bias_o[i] + (1 - this.beta2) * grad * grad;
            const m_hat = this.m_bias_o[i] / (1 - Math.pow(this.beta1, this.t));
            const v_hat = this.v_bias_o[i] / (1 - Math.pow(this.beta2, this.t));
            return bias + this.learning_rate * m_hat / (Math.sqrt(v_hat) + this.epsilon);
        });
  
        // Backpropagate the error to the fully connected layer
        let hidden_errors = this.weights_ho[0].map((_, i) =>
            this.weights_ho.reduce((sum, weights_row, j) => sum + weights_row[i] * output_errors[j], 0));
  
        // Calculate gradients for the fully connected layer
        let hidden_gradients = hidden.map((output, i) => dsigmoid(output) * hidden_errors[i]);
  
        // Gradient clipping for hidden gradients
        hidden_gradients = hidden_gradients.map(grad => Math.max(-this.gradient_clip, Math.min(this.gradient_clip, grad)));
  
        // Update weights and biases for the fully connected layer using Adam
        this.weights_fc.forEach((weights_row, i) => {
            weights_row.forEach((weight, j) => {
                const grad = hidden_gradients[i] * flattened[j];
                this.m_weights_fc[i][j] = this.beta1 * this.m_weights_fc[i][j] + (1 - this.beta1) * grad;
                this.v_weights_fc[i][j] = this.beta2 * this.v_weights_fc[i][j] + (1 - this.beta2) * grad * grad;
                const m_hat = this.m_weights_fc[i][j] / (1 - Math.pow(this.beta1, this.t));
                const v_hat = this.v_weights_fc[i][j] / (1 - Math.pow(this.beta2, this.t));
                this.weights_fc[i][j] += this.learning_rate * m_hat / (Math.sqrt(v_hat) + this.epsilon);
            });
        });
  
        this.bias_fc = this.bias_fc.map((bias, i) => {
            const grad = hidden_gradients[i];
            this.m_bias_fc[i] = this.beta1 * this.m_bias_fc[i] + (1 - this.beta1) * grad;
            this.v_bias_fc[i] = this.beta2 * this.v_bias_fc[i] + (1 - this.beta2) * grad * grad;
            const m_hat = this.m_bias_fc[i] / (1 - Math.pow(this.beta1, this.t));
            const v_hat = this.v_bias_fc[i] / (1 - Math.pow(this.beta2, this.t));
            return bias + this.learning_rate * m_hat / (Math.sqrt(v_hat) + this.epsilon);
        });
  
        // Calculate the loss (sum of absolute errors)
        return output_errors.reduce((sum, error) => sum + Math.abs(error), 0);
    }
  }
  
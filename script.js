// Utility functions (same as before)
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

// Neural Network class with a convolutional layer and optimizations
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
        this.learning_rate = 7e-3;
        this.momentum = 0.9;
        this.weight_decay = 1e-5;
        this.lr_decay = 0.99;

        // Velocity terms for momentum
        this.velocities_fc = this.weights_fc.map(row => row.map(() => 0));
        this.velocities_ho = this.weights_ho.map(row => row.map(() => 0));
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
        let output = this.weights_ho.map((weights_row, i) => 
            sigmoid(weights_row.reduce((sum, weight, j) => sum + weight * hidden[j], this.bias_o[i])));

        return output;
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

        // Calculate output errors
        let output_errors = outputs.map((output, i) => target_array[i] - output);

        // Calculate gradients for the output layer
        let gradients = outputs.map((output, i) => dsigmoid(output) * output_errors[i] * this.learning_rate);

        // Update weights and biases for the output layer with momentum and weight decay
        this.weights_ho.forEach((weights_row, i) => {
            weights_row.forEach((weight, j) => {
                const delta = gradients[i] * hidden[j];
                this.velocities_ho[i][j] = this.momentum * this.velocities_ho[i][j] + delta - this.weight_decay * weight;
                this.weights_ho[i][j] += this.velocities_ho[i][j];
            });
        });
        this.bias_o = this.bias_o.map((bias, i) => bias + gradients[i]);

        // Backpropagate the error to the fully connected layer
        let hidden_errors = this.weights_ho[0].map((_, i) =>
            this.weights_ho.reduce((sum, weights_row, j) => sum + weights_row[i] * output_errors[j], 0));

        // Calculate gradients for the fully connected layer
        let hidden_gradients = hidden.map((output, i) => dsigmoid(output) * hidden_errors[i] * this.learning_rate);

        // Update weights and biases for the fully connected layer with momentum and weight decay
        this.weights_fc.forEach((weights_row, i) => {
            weights_row.forEach((weight, j) => {
                const delta = hidden_gradients[i] * flattened[j];
                this.velocities_fc[i][j] = this.momentum * this.velocities_fc[i][j] + delta - this.weight_decay * weight;
                this.weights_fc[i][j] += this.velocities_fc[i][j];
            });
        });
        this.bias_fc = this.bias_fc.map((bias, i) => bias + hidden_gradients[i]);

        // Apply learning rate decay
        this.learning_rate *= this.lr_decay;

        // Calculate the loss (sum of absolute errors)
        return output_errors.reduce((sum, error) => sum + Math.abs(error), 0);
    }
}

// Visualization setup remains the same
const canvas = document.getElementById('visualizationCanvas');
const ctx = canvas.getContext('2d');const progressText = document.getElementById('progress');
const predictedLabelText = document.getElementById('predictedLabel');
const probabilitiesDiv = document.getElementById('probabilities');
const lossCanvas = document.getElementById('lossCanvas');
const lossCtx = lossCanvas.getContext('2d');

let losses = [];

function drawDigit(digitArray) {
    const imageData = ctx.createImageData(28, 28);
    for (let i = 0; i < digitArray.length; i++) {
        const color = digitArray[i] * 255;
        imageData.data[i * 4] = color;
        imageData.data[i * 4 + 1] = color;
        imageData.data[i * 4 + 2] = color;
        imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

function getRandomTrainingData() {
    const inputs = new Array(784).fill(0).map(() => Math.random());
    const label = Math.floor(Math.random() * 10);
    const targets = new Array(10).fill(0);
    targets[label] = 1;
    return { inputs, targets, label };
}

function updateProgress(percent) {
    progressText.textContent = `${percent.toFixed(2)}%`;
}

function updatePredictions(predictions) {
    const maxPrediction = Math.max(...predictions);
    const predictedLabel = predictions.indexOf(maxPrediction);
    predictedLabelText.textContent = predictedLabel;

    probabilitiesDiv.innerHTML = '';
    predictions.forEach((probability, i) => {
        const p = document.createElement('p');
        p.textContent = `Label ${i}: ${(probability * 100).toFixed(2)}%`;
        probabilitiesDiv.appendChild(p);
    });
}

function plotLoss(loss) {
    losses.push(loss);
    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

    lossCtx.beginPath();
    lossCtx.moveTo(0, lossCanvas.height);
    for (let i = 0; i < losses.length; i++) {
        const x = (i / (losses.length - 1)) * lossCanvas.width;
        const y = lossCanvas.height - (losses[i] * lossCanvas.height / Math.max(...losses));
        lossCtx.lineTo(x, y);
    }
    lossCtx.stroke();
}

// Training the CNN with Visualizatio n
const cnn = new ConvNet(28, 9, 36, 128, 10);  // Parameters: input_dim, kernel_dim, num_kernels, hidden_nodes, output_nodes
let trainingIterations = 170;

async function fetchAndProcessMnistData() {
    try {
        const response = await fetch("https://datasets-server.huggingface.co/rows?dataset=ylecun%2Fmnist&config=mnist&split=train&offset=0&length=100");
        const result = await response.json();

        // Inspect the structure of the result
        

        // Adjust the data extraction based on the actual structure
        // For example, if the actual MNIST data is nested in a specific property
        if (!result || !result.rows || !Array.isArray(result.rows)) {
            throw new Error("Expected data to be in 'rows' property as an array.");
        }

        // Process each row in the array
        const data = await Promise.all(result.rows.map(async (row) => {
            if (!row.row || !row.row.image || !row.row.image.src || row.row.label === undefined) {
                console.error("Invalid data format", row);
                return null;
            }

            const imageSrc = row.row.image.src;
            const label = row.row.label;

            try {
                const inputs = await extractPixelData(imageSrc);
                const targets = new Array(10).fill(0);
                targets[label] = 1;
                return { inputs, targets, label };
            } catch (error) {
                console.error('Error processing image:', error);
                return null;
            }
        }));

        // Filter out any null results
        return data.filter(item => item !== null);
    } catch (error) {
        console.error("Error fetching data:", error);
        return [];
    }
}

async function extractPixelData(imageSrc) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous'; // Handle CORS issues
        img.src = imageSrc;

        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            const imageData = ctx.getImageData(0, 0, img.width, img.height);
            const data = imageData.data;
            const pixels = [];

            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                const gray = (r + g + b) / 3 / 255; // Normalize to [0, 1]
                pixels.push(gray);
            }

            resolve(pixels);
        };

        img.onerror = (error) => reject(error);
    });
}

async function trainNetwork(epochs) {
    // Fetch and process the MNIST data
    const mnistData = await fetchAndProcessMnistData();
    const trainingIterations = mnistData.length;

    for (let epoch = 0; epoch < epochs; epoch++) {
        console.log(`Epoch ${epoch + 1}/${epochs}`);

        for (let i = 0; i < trainingIterations; i++) {
            const { inputs, targets, label } = mnistData[i];
            const loss = cnn.train(inputs, targets);
            console.log("iteration", i, "loss", loss);

            const predictions = cnn.predict(inputs);

            drawDigit(inputs);
            updateProgress(((i + 1) / trainingIterations) * 100);
            updatePredictions(predictions);
            plotLoss(loss);
        }

        // Optionally, you can add code here to evaluate the model on a validation set or print epoch summaries
    }
}

// Start the training process with a specified number of epochs
const numberOfEpochs = 16; // Set the number of epochs
trainNetwork(numberOfEpochs);

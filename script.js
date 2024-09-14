
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

function plotLoss(loss) {
  // Update the losses array
  losses.push(loss);
  
  // Clear the canvas for a new frame
  lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

  // Set up canvas styles for better visuals
  lossCtx.lineWidth = 2;           // Thicker line for better visibility
  lossCtx.strokeStyle = "#007bff"; // Blue line for loss curve
  lossCtx.fillStyle = "#f0f0f0";   // Background color for the chart
  lossCtx.font = "12px Arial";     // Font for labels
  lossCtx.textAlign = "center";    // Align text center

 




  // Calculate max loss for scaling
  const maxLoss = Math.max(...losses);

  // Draw the loss curve
  lossCtx.beginPath();
  lossCtx.strokeStyle = "#007bff";
  lossCtx.lineWidth = 2;
  lossCtx.moveTo(0, lossCanvas.height);

  for (let i = 0; i < losses.length; i++) {
      const x = (i / (losses.length - 1)) * lossCanvas.width;
      const y = lossCanvas.height - (losses[i] * lossCanvas.height / maxLoss);
      lossCtx.lineTo(x, y);
  }

  lossCtx.stroke();


  // Optionally: Display current loss as text
  const currentLoss = losses[losses.length - 1].toFixed(4);
  lossCtx.fillStyle = "#000";
  lossCtx.fillText(`Current Loss: ${currentLoss}`, lossCanvas.width / 2, 20);
}

function drawSoftmaxPredictions(predictions) {
  const canvas = document.getElementById('softmaxCanvas');
  const ctx = canvas.getContext('2d');

  // Clear the canvas before drawing
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Get the number of classes (length of the softmax predictions)
  const numClasses = predictions.length;

  // Define some variables for drawing
  const barWidth = canvas.width / numClasses;  // Width of each bar
  const maxBarHeight = canvas.height - 20;     // Maximum height for the bars, leave space for labels

  // Loop through predictions and draw each as a bar
  predictions.forEach((probability, index) => {
      const barHeight = probability * maxBarHeight;  // Height proportional to the softmax output
      const x = index * barWidth;                   // X position for the bar
      const y = canvas.height - barHeight;          // Y position (inverted to start from bottom)

      // Draw the bar
      ctx.fillStyle = '#007bff';                    // Bar color
      ctx.fillRect(x, y, barWidth - 5, barHeight);  // Draw the bar (leave space between bars)

      // Add the label below each bar (class index)
      ctx.fillStyle = '#000';
      ctx.textAlign = 'center';
      ctx.fillText(`${index}`, x + barWidth / 2, canvas.height - 5);

      // Add the probability value above each bar
      if(probability.toFixed(2) > 0.05){
          ctx.fillText(`${probability.toFixed(1) * 100}%`, x + barWidth / 2, y - 5);
      }
  });
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



const cnn = new ConvNet(Data.model_params.input_dim, Data.model_params.kernel_dim, Data.model_params.num_kernels, Data.model_params.hidden_nodes, Data.model_params.output_nodes); 
async function fetchAndProcessMnistData() {
    try {

        const labels = Data.training_data

        if (!Array.isArray(labels)) {
            throw new Error("Expected labels to be an array.");
        }

        // Process each label in the array
        const data = await Promise.all(labels.map(async (item) => {
            const { image: base64Image, label } = item;

            if (!base64Image || label === undefined) {
                console.error("Invalid data format", item);
                return null;
            }

            try {
                const inputs = await extractPixelData(base64Image);
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

async function extractPixelData(base64Image) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'Anonymous'; // Handle CORS issues
        img.src = base64Image;

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
const canvs = document.getElementById("networkCanvas");
const ct = canvs.getContext("2d");
const cans = document.getElementById("FeatureMaps");
const c = cans.getContext("2d");

// Neural network visualization parameters
const layerX = [0, 100, 400];  // X coordinates for layers
const nodeRadius = 20;
const inputDim = 28;  // For example, 28x28 input images
const numKernels = 8;
const kernelDim = 3;
const hiddenNodes = 10;
const outputNodes = 10;
const spaceY = 40;

// Example network parameters
let convNet = new ConvNet(inputDim, kernelDim, numKernels, hiddenNodes, outputNodes);

// Helper function to draw a circle for a node with gradient fill
function drawNode(x, y, activation, label = '') {
  const gradient = ct.createRadialGradient(x, y, nodeRadius * 0.6, x, y, nodeRadius);
  gradient.addColorStop(0, `rgba(0, 0, 255, ${activation})`);
  gradient.addColorStop(1, 'rgba(0, 0, 255, 0.1)');

  ct.beginPath();
  ct.arc(x, y, nodeRadius, 0, Math.PI * 2);
  ct.fillStyle = gradient;
  ct.fill();
  ct.strokeStyle = 'black';
  ct.lineWidth = 2;
  ct.stroke();

  if (label) {
    ct.fillStyle = "black";
    ct.font = "12px Arial";
    ct.fillText(label, x - 5, y + 5);
  }
}

// Helper function to draw a connection (weight) with arrowheads
function drawConnection(x1, y1, x2, y2, weight) {
  const thickness = Math.abs(weight) * 2;  // Weight thickness based on magnitude
  const arrowSize = 6;

  ct.beginPath();
  ct.moveTo(x1, y1);
  ct.lineTo(x2, y2);
  ct.lineWidth = thickness;
  ct.strokeStyle = "grey";
  ct.stroke();

  // Draw arrowhead
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ct.beginPath();
  ct.moveTo(x2, y2);
  ct.lineTo(x2 - arrowSize * Math.cos(angle - Math.PI / 6), y2 - arrowSize * Math.sin(angle - Math.PI / 6));
  ct.lineTo(x2 - arrowSize * Math.cos(angle + Math.PI / 6), y2 - arrowSize * Math.sin(angle + Math.PI / 6));
  ct.closePath();
  ct.fillStyle = ct.strokeStyle;
  ct.fill();
}

// Draw layers with labels
function drawNetwork(inputActivations, hiddenActivations, outputActivations, weights, biases) {
  ct.clearRect(0, 0, canvs.width, canvs.height);  // Clear the canvas



  // Draw hidden layer and connections
  for (let j = 0; j < hiddenNodes; j++) {
    let yHidden = 50 + j * spaceY;
    drawNode(layerX[1], yHidden, hiddenActivations[j] || 0, `H${j}`);

    
  }

  // Draw output layer and connections
  for (let k = 0; k < outputNodes; k++) {
    let yOutput = 50 + k * spaceY;
    drawNode(layerX[2], yOutput, outputActivations[k] || 0, `O${k}`);

    for (let j = 0; j < hiddenNodes; j++) {
      let yHidden = 50 + j * spaceY;
      drawConnection(layerX[1], yHidden, layerX[2], yOutput, weights.hiddenToOutput[j][k]);
    }
  }

  // Draw layer labels
  ct.fillStyle = "black";
  ct.font = "16px Arial";
  ct.fillText("Hidden Layer", layerX[1] - 30, 30);
  ct.fillText("Output Layer", layerX[2] - 30, 30);
}

function drawFeatureMaps(featureMapsData) {
  const mapWidth = featureMapsData.dimensions.width;
  const mapHeight = featureMapsData.dimensions.height;
  const padding = 10;    // Space between feature maps

  // Calculate the number of rows and columns
  const numMaps = featureMapsData.featureMaps.length;
  const cols = Math.ceil(Math.sqrt(numMaps));
  const rows = Math.ceil(numMaps / cols);

  // Draw each feature map as a grayscale image
  featureMapsData.featureMaps.forEach((featureMap, index) => {
    const x = 10 + (index % cols) * (mapWidth + padding);
    const y = 10 + Math.floor(index / cols) * (mapHeight + padding);

    // Normalize feature map
    const flatMap = featureMap.flat(); // Flatten 2D array to 1D
    const min = Math.min(...flatMap);
    const max = Math.max(...flatMap);
    const range = max - min;

    // Draw feature map
    for (let row = 0; row < mapHeight; row++) {
      for (let col = 0; col < mapWidth; col++) {
        const value = (featureMap[row][col] - min) / range;
        c.fillStyle = `rgba(${value * 255}, ${value * 255}, ${value * 255}, 1)`;
        c.fillRect(x + col, y + row, 1, 1);
      }
    }

    // Draw feature map label
    c.fillStyle = "black";
    c.font = "10px Arial";
    c.fillText(`F${index}`, x, y - 2);
  });
}


// Get the current activations and weights
function updateVisualization(inputs) {
  let inputArray = inputs;  // Example input
  let activations = cnn.getActivations(inputArray);
  let weightsAndBiases = cnn.getWeightsAndBiases(inputArray);
  let featureMaps = cnn.getFeatureMaps(inputArray); // Get feature maps for visualization


  drawNetwork(
    activations.input,
    activations.hidden,
    activations.output,
    weightsAndBiases,
    weightsAndBiases
  );

  drawFeatureMaps(featureMaps); // Draw feature maps
}


async function trainNetwork(epochs) {

  // Fetch and process the MNIST data
  const mnistData = await fetchAndProcessMnistData();
  const trainingIterations = mnistData.length;

  for (let epoch = 0; epoch < epochs; epoch++) {
      console.log(`Epoch ${epoch + 1}/${epochs}`);

      for (let i = 0; i < trainingIterations; i++) {
          // Fetch random data from the dataset
          const { inputs, targets, label } = mnistData[Math.floor(Math.random() * 1000)];
          
          // Perform the training step
          const loss = cnn.train(inputs, targets);

          // Plot the loss
          plotLoss(loss);

          console.log("Iteration", i, "Loss", loss);

          // Pause to allow UI to update
          await new Promise(resolve => setTimeout(resolve, 0)); // This allows the UI to update in between iterations

          // After 100 iterations, make a prediction
          if ((i + 1) % 100 === 0) {
              const randomIndex = Math.floor(Math.random() * trainingIterations);
              const { inputs: predInputs, label: predLabel } = mnistData[randomIndex];

              // Make a prediction
              const predictions = cnn.predict(predInputs);

              drawSoftmaxPredictions(predictions)
             

              // Draw the digit and update predictions in the UI
              drawDigit(predInputs);
             updateVisualization(predInputs);
              //updatePredictions(predictions);
              

              console.log(`Prediction after ${i + 1} iterations: ${predictions}`);
          }
      }
  }
}

trainNetwork(1)
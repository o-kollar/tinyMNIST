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

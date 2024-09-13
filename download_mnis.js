const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

// Path to the JSON file where labels and Base64 images will be saved
const LABELS_FILE = path.join(__dirname, 'labels.json');

// Append new labels to the labels JSON file
function appendLabelsToFile(newLabels) {
    let existingLabels = [];

    // Load existing labels from the file, if it exists
    if (fs.existsSync(LABELS_FILE)) {
        const fileContent = fs.readFileSync(LABELS_FILE);
        existingLabels = JSON.parse(fileContent);
    }

    // Append the new labels and save
    const updatedLabels = [...existingLabels, ...newLabels];
    fs.writeFileSync(LABELS_FILE, JSON.stringify(updatedLabels, null, 2));
    console.log(`Labels updated in ${LABELS_FILE}`);
}

// Fetch and process MNIST data for a given offset
async function fetchAndProcessMnistData(offset) {
    const apiUrl = `https://datasets-server.huggingface.co/rows?dataset=ylecun%2Fmnist&config=mnist&split=train&offset=${offset}&length=100`;

    try {
        const response = await fetch(apiUrl);
        const result = await response.json();

        if (!result || !result.rows || !Array.isArray(result.rows)) {
            throw new Error("Expected data to be in 'rows' property as an array.");
        }

        const labels = [];

        // Process each image
        for (const [index, row] of result.rows.entries()) {
            if (!row.row || !row.row.image || !row.row.image.src || row.row.label === undefined) {
                console.error("Invalid data format", row);
                continue;
            }

            const imageSrc = row.row.image.src;
            const label = row.row.label;

            try {
                // Convert the image to Base64 and add to labels array
                const base64Image = await getBase64Image(imageSrc);
                labels.push({ image: base64Image, label: label });
                console.log(`Processed image with label: ${label}`);
            } catch (error) {
                console.error('Error processing image:', error);
            }
        }

        // Append labels to a JSON file
        appendLabelsToFile(labels);

    } catch (error) {
        console.error("Error fetching data:", error);
    }
}

// Convert image URL to Base64
async function getBase64Image(imageUrl) {
    const response = await fetch(imageUrl);

    if (!response.ok) {
        throw new Error(`Failed to fetch image: ${imageUrl}`);
    }

    const buffer = await response.buffer();
    return `data:image/png;base64,${buffer.toString('base64')}`;
}

// Run the script to download 1,000 images in batches of 100
async function downloadAllImages() {
    const totalImages = 30000;
    const batchSize = 100;

    for (let offset = 0; offset < totalImages; offset += batchSize) {
        console.log(`Fetching images with offset: ${offset}`);
        await fetchAndProcessMnistData(offset);
    }

    console.log('Finished downloading all 1,000 MNIST images and labels.');
}

// Start the download process
downloadAllImages();

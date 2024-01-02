function predictPersonality() {
    var inputText = document.getElementById("inputText").value;

    // Replace this with your actual logic for making predictions
    // For demonstration purposes, let's assume a synchronous response
    var predictedPersonality = getPredictedPersonality(inputText);

    // Display the prediction result on the page
    var resultContainer = document.getElementById("predictionResult");
    resultContainer.innerHTML = "Predicted Personality Type: " + predictedPersonality;
}

function getPredictedPersonality(inputText) {
    // Replace this with the actual logic for making predictions
    // For now, let's assume a synchronous response
    return "ENFP";
}

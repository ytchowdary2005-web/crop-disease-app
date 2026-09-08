// Change this if your backend runs on a different host/port
const API_URL = "http://127.0.0.1:8000/predict";

const form = document.getElementById("predict-form");
const fileInput = document.getElementById("file");
const preview = document.getElementById("preview");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const submitBtn = document.getElementById("submit-btn");

// Show a preview of the uploaded image
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  errorBox.style.display = "none";
  resultBox.style.display = "none";

  const crop = document.getElementById("crop").value;
  const file = fileInput.files[0];

  if (!crop || !file) {
    showError("Please select a crop and upload an image.");
    return;
  }

  const formData = new FormData();
  formData.append("crop", crop);
  formData.append("file", file);

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || "Prediction failed.");
    }

    const data = await response.json();
    showResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Is the backend running?");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});

function showResult(data) {
  document.getElementById("predicted-class").textContent = data.predicted_class;
  document.getElementById("confidence").textContent = data.confidence;

  const allProbsDiv = document.getElementById("all-probs");
  allProbsDiv.innerHTML = "";
  Object.entries(data.all_probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, prob]) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `<span>${label}</span><span>${prob}%</span>`;
      allProbsDiv.appendChild(row);
    });

  resultBox.style.display = "block";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = "block";
}

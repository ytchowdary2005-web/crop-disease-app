# Crop Disease Detector

A full-stack app that serves 4 trained CNN models (chilli, finger millet/ragi,
rice, sugarcane) behind a FastAPI backend, with a simple HTML/JS frontend.

## Project structure
```
crop-disease-app/
├── backend/
│   ├── main.py            # FastAPI app, loads models, /predict endpoint
│   ├── requirements.txt
│   └── models/             # put your 4 .h5 files here
│       ├── chilli_model.h5
│       ├── finger_millet_model.h5
│       ├── rice_model.h5
│       └── sugarcane_model.h5
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

## Setup (in VS Code)

### 1. Open the project
Open the `crop-disease-app` folder in VS Code (`File > Open Folder`).

### 2. Add your model files
Copy your 4 trained `.h5` files into `backend/models/`, matching the
filenames in `main.py` (or edit `CROP_CONFIG` in `main.py` to match your
actual filenames).

### 3. Fill in the real class labels
Open `backend/main.py` and replace the placeholder `"class_0", "class_1"...`
lists in `CROP_CONFIG` with your actual disease names, **in the same order**
your model was trained on. You can get this from the `class_indices` you
printed in your training notebook, e.g.:
```python
train_generator.class_indices
# {'bacterial_blight': 0, 'blast': 1, 'healthy': 2, ...}
```
Sort by the index value and use that exact order.

### 4. Set up the backend
Open a terminal in VS Code (`` Ctrl+` ``) and run:
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Leave this terminal running. Visit `http://127.0.0.1:8000` in a browser —
you should see a JSON health-check response listing the 4 crops.

### 5. Run the frontend
Easiest option: install the **Live Server** extension in VS Code, right-click
`frontend/index.html`, and choose "Open with Live Server". It will open in
your browser (usually `http://127.0.0.1:5500`).

(Alternatively, just double-click `index.html` to open it directly in a
browser — it will still work since it calls the backend via `fetch`.)

### 6. Test it
Select a crop, upload a leaf image, click Predict. You should see the
predicted disease class and confidence score.


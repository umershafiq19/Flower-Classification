================================================================================
    MLflow + DagsHub + Flask Lab - Complete Setup Guide
================================================================================

PREREQUISITES
-------------
- Python 3.8 or higher
- VS Code installed
- Internet connection

================================================================================
STEP-BY-STEP SETUP
================================================================================

STEP 1: CREATE PROJECT FOLDER
------------------------------
Create a folder named: mlops-assignment


STEP 2: ADD PROJECT FILES
--------------------------
Put these 4 files in your folder:

mlops-assignment/
├── train.py
├── app.py
├── requirements.txt
└── templates/
    └── upload.html


STEP 3: OPEN IN VS CODE
------------------------
1. Open VS Code
2. File → Open Folder
3. Select mlops-assignment folder


STEP 4: CREATE VIRTUAL ENVIRONMENT
-----------------------------------
1. Press Ctrl+Shift+P (Command Palette)
2. Type: Python: Create Environment
3. Select: Venv
4. Select your Python interpreter
5. Wait for venv to be created


STEP 5: INSTALL DEPENDENCIES
-----------------------------
Open terminal in VS Code (Ctrl+`) and run:

    pip install -r requirements.txt

This will take 5-10 minutes.


================================================================================
STEP 6: SETUP DAGSHUB
================================================================================

6.1 CREATE ACCOUNT
------------------
1. Go to https://dagshub.com
2. Sign up with email or GitHub


6.2 CREATE REPOSITORY
---------------------
1. Click "New Repository"
2. Name: mlflow-flask-assignment
3. Make it PUBLIC
4. Click "Create Repository"


6.3 GET ACCESS TOKEN
--------------------
1. Go to: https://dagshub.com/user/settings/tokens
2. Click "Create New Token"
3. Name it: "MLflow Lab"
4. Click "Create"
5. COPY THE TOKEN (you won't see it again!)


6.4 SETUP GIT PASSWORD (FIRST TIME ONLY)
-----------------------------------------
1. Go to: https://dagshub.com/user/settings
2. Click "Password" tab
3. Set a password for Git operations
4. Remember this password!


6.5 UPDATE CREDENTIALS IN CODE
-------------------------------
Edit train.py (lines 18-20):

    DAGSHUB_USERNAME = "your_actual_username"
    DAGSHUB_REPO = "mlflow-flask-assignment"
    DAGSHUB_TOKEN = "paste_your_token_here"

Edit app.py (lines 17-19):

    DAGSHUB_USERNAME = "your_actual_username"
    DAGSHUB_REPO = "mlflow-flask-assignment"
    DAGSHUB_TOKEN = "paste_your_token_here"


================================================================================
STEP 7: TRAIN MODEL
================================================================================

Run in terminal:

    python train.py

When asked: Continue with local MLflow? (y/n):
Type: y

EXPECTED:
- Dataset downloads (~2-5 min)
- Training runs (10-15 min)
- Should see: Found 5 classes: ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
- Final accuracy: 60-75%


================================================================================
STEP 8: REGISTER MODEL IN DAGSHUB
================================================================================

8.1 GO TO DAGSHUB
-----------------
Open: https://dagshub.com/YOUR_USERNAME/mlflow-flask-assignment


8.2 NAVIGATE TO EXPERIMENTS
----------------------------
1. Click "Experiments" tab
2. You'll see your training run


8.3 REGISTER MODEL
------------------
1. Click on your run
2. Scroll to "Artifacts" section
3. Click "model" folder
4. Click "Register Model" button
5. Keep name as: flowers_classifier
6. Click "Register"


8.4 SET TO PRODUCTION
---------------------
1. Go to "Models" tab (top navigation)
2. Click on flowers_classifier
3. You'll see Version 1
4. Click the "Stage" dropdown
5. Select "Production"
6. Confirm


8.5 TAKE SCREENSHOT
-------------------
Take a screenshot showing:
- Model name: flowers_classifier
- Version: 1
- Stage: Production

Save as: model_registry_screenshot.png


================================================================================
STEP 9: RUN FLASK APP
================================================================================

In terminal:

    python app.py

EXPECTED OUTPUT:
    Starting Flask App...
    ✅ Model loaded from MLflow Registry!
    App running at: http://localhost:5000

Open browser: http://localhost:5000

TEST THE APP:
1. Upload a flower image
2. Click "Predict"
3. See prediction results!


================================================================================
STEP 10: RECORD DEMO VIDEO
================================================================================

WHAT TO RECORD (1-2 minutes):
1. Terminal showing Flask app running
2. Browser at localhost:5000
3. Upload first flower image
4. Show prediction results
5. Upload second different flower
6. Show second prediction

Save as: flask_demo.mp4


================================================================================
STEP 11: PUSH CODE TO DAGSHUB
================================================================================

In terminal:

    git init
    git add .
    git commit -m "Complete MLflow Flask Assignment"
    git remote add origin https://dagshub.com/<YOUR_USERNAME>/mlflow-flask-flowers.git
    git branch -M main
    git push -u origin main


IF YOU GET "HTTP 500" ERROR OR "REMOTE HUNG UP":
-------------------------------------------------
This happens because the trained model file (*.keras) is too large for Git.

REASONS:
- Model files are 80-100MB in size
- Git is designed for code, not large binary files
- DagsHub has upload limits for HTTP push
- The model is already stored in MLflow Registry (no need in Git)

SOLUTION - Remove model from Git and push again:

    git rm --cached *.keras
    git commit -m "Remove model file"
    git push -u origin main

This removes the model from Git tracking but keeps your code.
The model is safely stored in DagsHub's MLflow Registry.


WHEN PROMPTED FOR CREDENTIALS:
-------------------------------
- Username: your DagsHub username
- Password: your DagsHub Git password (from Step 6.4, NOT your login password)


================================================================================
STEP 12: SUBMIT
================================================================================

You need to submit:

1. DAGSHUB REPOSITORY URL
   https://dagshub.com/YOUR_USERNAME/mlflow-flask-assignment

2. MODEL REGISTRY SCREENSHOT
   File: model_registry_screenshot.png

3. DEMO VIDEO
   File: flask_demo.mp4


================================================================================
VERIFICATION CHECKLIST
================================================================================

Before submitting, check:
□ Model trained successfully (60-75% accuracy)
□ Experiments visible on DagsHub
□ Model registered as "flowers_classifier"
□ Model stage is "Production"
□ Screenshot shows registered model
□ Flask app works and makes predictions
□ Demo video recorded
□ All code pushed to DagsHub
□ Repository is PUBLIC


================================================================================
TROUBLESHOOTING
================================================================================

DATASET SHOWS "1 CLASSES" INSTEAD OF 5:
---------------------------------------
Clear cache and run training again:

    Remove-Item -Recurse -Force $env:USERPROFILE\.keras\datasets\flower_photos
    python train.py


MODEL WON'T LOAD IN FLASK:
--------------------------
- Check model is in "Production" stage
- Verify credentials in app.py
- Ensure model name is exactly "flowers_classifier"


GIT PUSH FAILS:
--------------
- Check you set Git password in DagsHub settings
- Use your DagsHub Git password (not account password)
- Check repository URL is correct
- If model file too large, remove it (see Step 11)


PORT 5000 IN USE:
-----------------
Edit app.py, change last line:
    app.run(debug=True, host='0.0.0.0', port=5001)


GIT REMOTE ALREADY EXISTS:
--------------------------
    git remote remove origin
    git remote add origin https://dagshub.com/USERNAME/mlflow-flask-assignment.git


================================================================================
EXPECTED RESULTS
================================================================================

- Training Accuracy: 65-85%
- Validation Accuracy: 60-75%
- Training Time: 10-15 minutes
- 5 Flower Classes: dandelion, daisy, tulips, sunflowers, roses


================================================================================
QUICK COMMAND REFERENCE
================================================================================

CREATE VENV:
    Ctrl+Shift+P → Python: Create Environment

INSTALL PACKAGES:
    pip install -r requirements.txt

TRAIN MODEL:
    python train.py

RUN FLASK APP:
    python app.py

GIT COMMANDS:
    git init
    git add .
    git commit -m "Complete assignment"
    git remote add origin https://dagshub.com/USERNAME/mlflow-flask-assignment.git
    git branch -M main
    git push -u origin main

IF MODEL FILE TOO LARGE:
    git rm --cached *.keras
    git commit -m "Remove model file"
    git push -u origin main


================================================================================
IMPORTANT NOTES
================================================================================

1. ALWAYS update credentials in both train.py and app.py
2. Repository MUST be PUBLIC for grading
3. Model stage MUST be "Production" for Flask to load it
4. Git password is different from DagsHub login password
5. Training takes time - be patient!
6. Model files should NOT be pushed to Git (they're in MLflow Registry)
7. If Git push fails with HTTP 500, remove *.keras files and push again


================================================================================
WHY REMOVE MODEL FILES FROM GIT?
================================================================================

Git is Version Control for CODE, not for LARGE FILES:
- Code files: train.py (10KB) ✓ Perfect for Git
- Model files: *.keras (80-100MB) ✗ Too large for Git

REASONS TO EXCLUDE MODEL FILES:
1. Size: Model files are 80-100MB, Git is slow with large files
2. Binary: Models are binary files, Git can't show useful diffs
3. Limits: DagsHub HTTP push has size limits (~100MB)
4. Redundant: Model is already in MLflow Registry on DagsHub
5. Best Practice: Use Git for code, MLflow for models

WHERE IS YOUR MODEL STORED?
- Git Repository: Code only (train.py, app.py)
- MLflow Registry: Trained model (accessible via app.py)

PROPER WORKFLOW:
1. Code → Git (train.py, app.py)
2. Model → MLflow Registry (flowers_classifier)
3. Flask loads model from MLflow, not from Git


================================================================================

TOTAL TIME: ~60-90 minutes
DIFFICULTY: Intermediate

Good luck! 🚀

================================================================================

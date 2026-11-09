# GitHub Step-by-Step Guide (from scratch)

## Part A - One-time setup
1. **Create a GitHub account** at https://github.com/signup and verify your email.
2. **Install Git**: https://git-scm.com/downloads (accept defaults). Check it worked: open Command Prompt / Terminal and run `git --version`.
3. **Install Python 3.10+**: https://www.python.org/downloads (tick "Add Python to PATH" on Windows). Check: `python --version`.
4. **Install VS Code** (optional but recommended): https://code.visualstudio.com
5. **Tell Git who you are** (once):
   ```
   git config --global user.name "Your Name"
   git config --global user.email "your-github-email@example.com"
   ```

## Part B - Create the repository on GitHub
1. Log in to GitHub, click the **+** (top right) -> **New repository**.
2. Repository name: `fake-clone-twitter-detection`
3. Description: `Detection of fake and clone Twitter accounts using classification and Euclidean distance`
4. Choose **Public** (so recruiters can see it).
5. Tick **Add a README file** and choose `.gitignore` template **Python**. Click **Create repository**.

## Part C - Get the repo onto your computer
```
cd Desktop
git clone https://github.com/<your-username>/fake-clone-twitter-detection.git
cd fake-clone-twitter-detection
```
(`clone` = download a copy that stays linked to GitHub.)

## Part D - Add the project files
1. Copy the files from this project (`src/`, `docs/`, `requirements.txt`, `README.md`, `.gitignore`) into the cloned folder, replacing the placeholder README and .gitignore.
2. Download the Kaggle dataset (https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset -> **Download**), unzip, and save the CSV as `data/twitter_bot_detection.csv`. (Kaggle needs a free account.)
3. Create a virtual environment and install libraries:
   ```
   python -m venv venv
   venv\Scripts\activate        (Mac/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   ```
4. Run the project:
   ```
   python src/pipeline.py
   ```
   If you get "Columns not found", open the CSV, compare the column names with the `COL` dictionary at the top of `src/pipeline.py`, and edit it to match.
5. Check the `outputs/` folder: charts, KPI file, model comparison, clone pairs.

## Part E - Save your work to GitHub (the 3 commands you will use forever)
```
git add .
git commit -m "Add analysis pipeline, outputs and documentation"
git push
```
- `add` = choose the changes, `commit` = save a snapshot with a message, `push` = upload to GitHub.
- First push may ask you to sign in through the browser. Do that once.

## Part F - Good commit habit (shows a real work history)
Commit after each stage with a clear message:
1. `Add project README and folder structure`
2. `Add data cleaning and quality report`
3. `Add EDA charts`
4. `Add feature engineering and model comparison`
5. `Add Euclidean-distance clone detection`
6. `Add KPI summary and BA documentation`
7. `Add dashboard screenshots`

## Part G - Make the repository look professional
1. Open the repo on GitHub -> check that README renders properly.
2. Put 3-4 chart PNGs and a dashboard screenshot in the README using:
   `![Model comparison](outputs/charts/07_model_comparison.png)`
3. Fill the "Results" section of README with your real numbers.
4. Click the gear icon next to **About** -> add topics: `data-analysis`, `python`, `scikit-learn`, `business-analyst`, `fake-account-detection`.
5. Pin the repo to your profile (Profile -> Customize your pins).
6. Add the repo link to your resume and LinkedIn.

## Part H - Common problems
| Problem | Fix |
|---|---|
| `git is not recognized` | Reinstall Git, restart terminal |
| `python is not recognized` | Reinstall Python with "Add to PATH" ticked |
| Push rejected / asks for password | Sign in through the browser popup, or use a Personal Access Token (GitHub -> Settings -> Developer settings -> Tokens) |
| CSV too large to push | Keep it in `.gitignore`; tell readers to download from Kaggle (already done) |
| `ModuleNotFoundError` | Activate the venv and run `pip install -r requirements.txt` |

## Alternative without command line
On GitHub, open your repo -> **Add file -> Upload files**, drag in your folders, write a commit message, click **Commit changes**. Good for a first try, but learn the 3 commands above.

# Chapter 0: Level 0 Orientation — Landing on Planet Regression

Hey there! Welcome. If you've ever looked at a messy spreadsheet and thought, *"I wonder if we can predict one column using the others without losing my mind,"* you're in the right place. 

This guide is designed to take you from a complete beginner—someone who doesn't know their features from their targets—to someone who can confidently sit down with PhD researchers, discuss causal estimation, and not break a sweat. All in about a week of focused work.

But we have a strict rule here: **no passive reading**. You don't learn swimming by reading a manual on hydrodynamics; you learn it by jumping into the pool and swallowing a bit of water. In our case, the pool is full of data, and the water is Python code. 

Let's get oriented.

---

## 1. How to Use This Guide

This is a self-paced, project-based curriculum. Instead of using toy datasets like Boston Housing (which has been beaten to death and has some pretty sketchy history anyway) or Iris flowers, we are going to work with three custom-built datasets based on Khyber Pakhtunkhwa (KP), Pakistan. 

For every single chapter, you should follow this rhythm:
1. **Read the theory** (informal, conceptual, but mathematically honest).
2. **Explain it back** (tell a rubber duck or your cat what the concept means).
3. **Run the experiments** (write the code yourself—no copy-pasting allowed).
4. **Break it on purpose** (what happens if you scale the target? what if you pass ID columns? find out!).
5. **Solve the exercises** (thoughtful, structured problems designed to build muscle memory).

### Your Toolkit
Make sure you have a working Python installation. We'll be using:
- `pandas` & `numpy` (the bread and butter of data manipulation)
- `matplotlib` & `seaborn` (for making plots that actually look good)
- `scikit-learn` (for machine learning)
- `statsmodels` (for statistical tests and classical econometrics)
- Advanced tools (which we will introduce as we go: `XGBoost`, `Optuna`, `SHAP`, `MAPIE`, etc.)

---

## 2. Setup and Data Generation

Suppose someone collected data across the beautiful valleys and plains of KP. Before we can do any math or fit any lines, we need to generate this data. 

### Step 2.1: The Command Line Ritual
Open your terminal and run the following commands to set up a clean, isolated environment. Don't install packages globally; that's how you break other projects and end up crying over dependency conflicts at 2 AM.

```bash
# Clone or navigate to your workspace
cd /var/www/documentation/regression-tasks

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the essentials
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Step 2.2: Generate the Synthetic KP Datasets
The project directory contains a script called [generate_data.py](file:///var/www/documentation/regression-tasks/generate_data.py). This script simulates a mini-world with realistic statistical dependencies. Let's run it:

```bash
python generate_data.py
```

If successful, you will see output like this:
```text
Generating development dataset: data/kp_subdistrict_development_index.csv with 100000 records...
Finished generating data/kp_subdistrict_development_index.csv.
Generating agricultural dataset: data/kp_agricultural_yields.csv with 100000 records...
Finished generating data/kp_agricultural_yields.csv.
Generating infrastructure dataset: data/kp_infrastructure_projects.csv with 100000 records...
Finished generating data/kp_infrastructure_projects.csv.
All datasets generated successfully in the 'data' directory!
```

> [!WARNING]
> These datasets are **synthetic**. They are mathematically designed to teach you regression. They reflect realistic terrain constraints, crop scales, and budget overruns of Pakistan, but they are **not** real survey data. Do not present these as actual policy evidence to the KP Planning and Development Department unless you want to get laughed out of the room.

---

## 3. The Three KP Regression Datasets

Let's meet our three main characters. Each represents a distinct style of regression problem.

### 3.1 The Development Index Dataset
* **File:** `data/kp_subdistrict_development_index.csv`
* **Target (What we want to find):** `development_score` (a bounded index from 0 to 100).
* **Concept:** *Suppose someone collected survey data from 100,000 communities in KP. They have indicators like literacy rate, school enrollment, average household income, distance to health facilities, and public funding. Now, they want to predict the overall community development score. What do we do?*
* **The Reality Inside:** This dataset is highly linear with some terrain-based penalties (e.g., mountainous regions face structural challenges). However, it contains severe **multicollinearity** (literacy rate and school enrollment are almost copies of each other).

### 3.2 The Agricultural Yields Dataset
* **File:** `data/kp_agricultural_yields.csv`
* **Target:** `crop_yield_tons_per_acre` (a positive continuous value).
* **Concept:** *Suppose a researcher collected farm-level data including crop types (wheat, maize, sugarcane, apples, peaches), soil pH, rainfall, elevation, fertilizer application, and whether the farm is organic. They want to predict the yield per acre to advise farmers. What do we do?*
* **The Reality Inside:** This dataset is highly **nonlinear** and filled with **interactions**. Soil pH has an optimal point (curves downwards if pH is too acidic or basic). Apples and peaches die in high temperatures, while sugarcane thrives. Organic farms have strictly zero fertilizer and pesticides. A simple straight line will fail miserably here.

### 3.3 The Infrastructure Projects Dataset
* **File:** `data/kp_infrastructure_projects.csv`
* **Targets:** `actual_cost_million_pkr` and `actual_duration_months` (two continuous values!).
* **Concept:** *Suppose the government of KP tracked 100,000 public works projects (roads, schools, water supply). They have the approved budget, estimated duration, contractor experience, and year of initiation. They want to predict the actual final cost and duration before building starts to prevent budget overruns. What do we do?*
* **The Reality Inside:** The targets are heavily **right-skewed** (most projects are small, but a few mega-projects cost billions). It is a **multi-output** problem where cost and duration are intimately related, but actual duration is unknown when planning.

---

## Chapter 0 Exercises: A Dress Rehearsal, Not a Preview

Here are your first hands-on challenges. Write python scripts (save them under `scratch/` or run them in an interactive environment) to solve them.

> **What this rehearsal is for.** These three datasets are large, realistic, and deliberately disconnected from the running case the rest of the book uses. That's on purpose. Chapter 1 onward follows a single narrower thread — the MHP Cost Estimator — built from small, hand-inspectable datasets that grow chapter by chapter, not from these CSVs. You will not see `kp_subdistrict_development_index.csv`, `kp_agricultural_yields.csv`, or `kp_infrastructure_projects.csv` again after this chapter, and that's fine. The point here is to build the reflex — load it, check its shape, find the ID columns, read the generating code — on data with real scale and real messiness, *before* the stakes go up and that reflex has to be automatic. Think of it as a practice scrimmage before the season starts, not the season itself.

### Exercise 0.1: The Data Handshake
Write a Python script `verify_setup.py` to load all three datasets and print their shapes, columns, and target variable ranges (minimum and maximum values).

**Helpful Comments:**
* Use `pd.read_csv()` to load.
* Use `.shape`, `.columns`, and `.describe()` to extract statistics.
* *Comic Relief:* If your script throws a `FileNotFoundError`, verify you ran `generate_data.py` first. Data does not materialize out of thin air, no matter how hard you stare at the screen.

### Exercise 0.2: Spotting the Imposters (Identifiers)
Look closely at the columns of all three datasets. Identify the columns that represent unique IDs (identifiers). 
1. Why must we drop these columns before feeding them to a regression model?
2. What happens if a model "memorizes" `community_id` to predict `development_score`?

**Helpful Comments:**
* Unique identifiers have high cardinality (almost every row has a unique value).
* If you train a model with ID columns, it might find a spurious pattern (e.g., ID `COM-00042` has a high score just by luck) and fail when predicting on a new community `COM-99999`. This is the ultimate form of cheating and overfitting.

### Exercise 0.3: Tracing the Data-Generating Code
Open [generate_data.py](file:///var/www/documentation/regression-tasks/generate_data.py) and search for the function `generate_development_dataset`. Look at how `raw_score` is calculated.
Write down the exact formula used to build `raw_score`.
What weights are given to `literacy_rate` vs `public_funding_allocated_million_pkr`?

---

## Handing Off to Chapter 1

One last thing before you turn the page. Everything in this chapter — the three CSVs, `generate_data.py`, the exercises above — belongs to this chapter alone. Chapter 1 opens a new, separate running case: the **MHP Cost Estimator**, predicting the cost of microhydro power projects from a handful of hand-built rows (eight, to start). It does not load anything you generated here, and it doesn't need to. Different dataset, same rules: read the theory, run the code yourself, break it on purpose. If Chapter 1 feels like a fresh start rather than a continuation, that's expected — you're not missing a step.

---

## References

1. **Python Virtual Environments:** [Python venv Docs](https://docs.python.org/3/library/venv.html) - Read this if you still don't know why we use virtual environments.
2. **pandas Basics:** [pandas Getting Started Guide](https://pandas.pydata.org/docs/getting_started/index.html) - Essential reading for data loading.
3. **Pakistan Development Context:** Planning & Development Department, Khyber Pakhtunkhwa [KP P&D Website](https://pndkp.gov.pk/) - For background on how development funding and infrastructure projects are managed in KP.

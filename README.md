JEE Trend Predictor: Hierarchical AI 🎯

A high-precision machine learning pipeline designed to predict topic recurrence in the Joint Entrance Examination (JEE). By treating $LaTeX$ mathematical structures as "structural fingerprints," the system achieves a 3.3x statistical edge over random topic distribution.

🚀 Overview

The JEE Trend Predictor moves beyond simple keyword matching. It analyzes the conceptual density of questions by mining $LaTeX$ source code, using a Two-Tier Stacking Architecture to resolve subject overlaps and filter for the most probable 2026 examination trends.

Key Performance Indicators (KPIs)

Precision: ~40% (Current) vs. ~12% (Baseline guessing).

Architecture: Hierarchical Multi-Model Stacking.

Data Source: Historical JEE Question Bank ($LaTeX$ format).

🏗️ Technical Architecture

1. Data Engineering & Extraction (jee_extractor.py)

The system treats $LaTeX$ as the primary data source, extracting "Structural Fingerprints" to define question complexity:

symbol_count: Measures conceptual density via backslash command frequency (e.g., \frac, \int, \sqrt).

Mathematical Identifiers: Binary features for has_integral, has_vec, and has_diff.

Is_Numerical: Distinguishes between standard MCQs and Integer-type questions based on structural cues.

Cleaning: Uses LatexNodes2Text for human-readable display while retaining raw code for the ML models to "read."

2. Hierarchical AI Design (model_builder_v3.py)

Tier 0: The Subject Experts

Independent specialists trained on subject-specific statistical rules:

Physics (SVM): Optimized for high-dimensional margins and unit-based density.

Chemistry (Random Forest): Optimized for categorical rules and subtopic recurrence.

Math (XGBoost): Optimized for complex, nested temporal trends in calculus and algebra.

Tier 1: Dual-Signal Heuristics

KNN Cosine Similarity (The Template Hunter): Identifies if a 2026 candidate is mathematically similar in structure to historical repeats.

Naive Bayes (The Purity Scorer): Measures the "purity" of subtopic language to identify hybrid vs. core concepts.

Tier 2: The Master Manager (Meta-Model)

A Logistic Regression model acts as the final decision-maker. It evaluates the Confidence Scores of Tier 0 and Tier 1 to determine the most reliable prediction, learning which "Expert" to trust based on the specific feature input.

📊 The "JEE Blueprint" Constraint

To prevent over-prediction and align with real-world exam formats, the system applies a Top-K Filter:

Fixed Volume: Forces the model to select exactly 30 topics per subject (matching the 90-question JEE paper format).

Precision Boost: This constraint mathematically eliminates "low-confidence guesses," ensuring only the most probable candidates are flagged.

⚠️ Challenges & Future Roadmap

The OCR Bottleneck

Current Issue: New 2026 papers are often images/PDFs. Standard OCR corrupts mathematical symbols (e.g., $\int$ becomes f), "blinding" the model’s structural analysis.

Solution: Future iterations will integrate a Vision Transformer (ViT) to accurately translate images back into the $LaTeX$ features the model understands.

🛠️ Project Structure

jee_extractor.py: Logic for $LaTeX$ mining and structural feature engineering.

model_builder_v3.py: Tiered stacking architecture implementation.

project_report.md: Deep dive into limitations, data cleaning, and KPI analysis.

Developed By

Harshit Gupta 

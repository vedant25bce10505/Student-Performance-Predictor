# Problem Statement

## Student Performance Predictor — ML Project

**Institution:** VIT Bhopal University  
**School:** Computing Science and Engineering  
**Author:** Vedant Tiwari  
**Academic Year:** 2025–2026

---

## Background

Student academic failure and dropout is a persistent challenge in higher education. Research shows that early identification of at-risk students significantly improves intervention outcomes, yet most institutions still rely on teacher intuition rather than data-driven signals — an approach that does not scale to large cohorts.

## Problem

Given a student's demographic, social, and academic profile, can a machine learning model accurately predict their final grade category (A / B / C / D / F) or pass/fail outcome early enough to enable meaningful intervention?

## Research Questions

1. Which features (study hours, attendance, parental background, etc.) are most predictive of academic performance?
2. Which ML algorithm — Random Forest, Gradient Boosting, Decision Tree, or Neural Network — achieves the best balance of accuracy and interpretability for this task?
3. Can the solution be packaged as a usable tool that educators can access through a web interface?
4. How can class imbalance (majority "Pass" labels) be handled without degrading recall on the minority "Fail" class?

## Objectives

- Build a preprocessing pipeline that handles missing values, outliers, and encoding robustly.
- Engineer informative composite features from raw inputs.
- Train and compare four ML algorithms using 5-fold cross-validation.
- Apply SMOTE to correct class imbalance.
- Deploy the best model as a Flask web application with persistent prediction storage.
- Achieve at least 85% weighted F1-score on the held-out test set.

## Scope

- Dataset: 10,000+ synthetic student records, 25+ features.
- Target: Grade category (A–F) derived from final exam scores.
- Deployment: Local Flask server; cloud deployment is a future extension.
- Explainability: Feature importance via tree models; SHAP integration is future work.

## Success Criteria

| Metric | Target |
|--------|--------|
| Weighted F1-Score | ≥ 85% |
| Recall on Fail class | ≥ 80% |
| Web app uptime (local) | 100% |
| Test coverage | ≥ 70% |

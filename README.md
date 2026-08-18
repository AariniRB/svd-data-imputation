# SVD Data Imputation

This project implements missing value imputation using **Singular Value Decomposition (SVD)** and **Null Space Linear Equations**. 

## How It Works
1. Detects low-rank linear relationships between dataset features using SVD.
2. Extracts null space vectors to construct precise linear equations ($Ax = b$).
3. Solves the system of equations dynamically to fill in missing (`NaN`) values without relying on simple averages or medians.

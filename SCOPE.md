# 📌 Project Scope: CCTV Violence Detection System

## Overview

This project aims to build a deep learning system that detects violent behavior in CCTV surveillance footage. The system classifies video segments into two categories: **violence** and **non-violence**.

---

## Objectives

- Detect fights and aggressive interactions
- Minimize false positives
- Handle real-world CCTV challenges (low quality, blur, lighting changes)

---

## Dataset

- 1000 violence videos
- 1000 non-violence videos
- Videos will be converted into frame sequences

---

## Approach

### 1. Preprocessing

- Extract frames from videos
- Normalize and standardize inputs
- Convert frames into sequences

### 2. Feature Learning

- Learn spatial (appearance) + temporal (motion) patterns
- Focus on human interaction, not just visuals

### 3. Modeling

- Start with:
  - CNN + LSTM
- Later improvements:
  - 3D CNN
  - Transformers

### 4. Evaluation

- Precision / Recall
- F1-score (important for violence detection)

---

## Challenges

- Violence is temporal, not always visible in one frame
- Noisy labels inside videos
- Dataset bias (lighting, angles, locations)
- Low-quality CCTV footage

---

## Expected Outcome

A model that:

- Analyzes short video clips
- Detects violent behavior
- Will work on CCTV real-time systems

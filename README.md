
# 🌿 AI for Biodiversity: Fine-Grained Monkey Species Classification
## 🚀 Neural Networks & Computer Vision (NNCV) Project Showcase

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 1. Project Vision & Business Imperative

Manual wildlife tracking across vast, remote conservation sanctuaries is an operational bottleneck for global forestry and preservation efforts. Traditional camera traps capture thousands of motion-activated images daily, creating a massive "data deluge" that requires weeks of manual human sorting. 

This repository implements a production-grade **Discriminative AI System** designed to solve this crisis. By modeling the conditional probability distribution $P(Y \mid X)$—predicting the target monkey species $Y$ given a raw visual pixel tensor $X$—this pipeline automates wildlife monitoring at scale.

```text
┌──────────────────────────────┐     ┌──────────────────────────────┐
│     Manual Patrol Limits     │  ►  │    Delayed Interventions     │
├──────────────────────────────┤     ├──────────────────────────────┤
│ Rangers cannot cover vast,   │     │ Cannot deploy medical or     │
│ remote terrains 24/7.        │     │ anti-poaching teams in time. │
└──────────────────────────────┘     └──────────────────────────────┘
                                  ▲
                                  │ (Automated Telemetry Bridges the Gap)
                                  │
┌──────────────────────────────┐     ┌──────────────────────────────┐
│    Static Camera Traps       │  ►  │    Data Deluge Bottleneck    │
├──────────────────────────────┤     ├──────────────────────────────┤
│ Thousands of motion photos   │     │ Humans take weeks to manually│
│ are captured every day.      │     │ sort empty leaves from monkeys│
└──────────────────────────────┘     └──────────────────────────────┘


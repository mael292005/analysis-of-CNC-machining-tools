# Tool Wear Prediction — Dataset Documentation

> **Project Goal:** Predict the wear state of a cutting tool (milling cutter / turning insert)
> from sensor signals, using machine learning and signal processing techniques.

---

## Table of Contents

1. [Dataset Overview & Links](#1-dataset-overview--links)
2. [Dataset 1 — PHM 2010 Challenge](#2-dataset-1--phm-2010-challenge)
3. [Dataset 2 — CNC Machining Energy (Brillinger)](#3-dataset-2--cnc-machining-energy-brillinger)
4. [Dataset 3 — CNC Turning Roughness & Tool Wear (adorigueto)](#4-dataset-3--cnc-turning-roughness--tool-wear-adorigueto)
5. [Cross-Dataset Comparison](#5-cross-dataset-comparison)
6. [Usage Recommendations](#6-usage-recommendations)
7. [Analysis Notebooks Index](#7-analysis-notebooks-index)

---

## 1. Dataset Overview & Links

| # | Name | URL | Format | License |
|---|---|---|---|---|
| 1 | PHM 2010 Challenge (Kaggle) | `https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010` | `.csv` | CC0-1.0 |
| 1 | PHM 2010 (IEEE DataPort) | `https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge` | `.csv` | IEEE |
| 2 | CNC Machining Energy (Mendeley) | `https://data.mendeley.com/datasets/gtvvwmz7r7/2` | `.json`, `.xlsx`, `.stl` | CC BY-NC 4.0 |
| 2 | Related PMC article | `https://pmc.ncbi.nlm.nih.gov/articles/PMC12269468/` | — | — |
| 3 | CNC Turning / adorigueto (Kaggle) | `https://www.kaggle.com/datasets/adorigueto/cnc-turning-roughness-forces-and-tool-wear` | `.csv` | Open |
| 3 | Author GitHub | `https://github.com/adorigueto/Prediction-with-ANN` | — | — |

### Quick Comparison

| Criterion | PHM 2010 | CNC Machining Energy | CNC Turning |
|---|---|---|---|
| **Process type** | Milling | 5-axis milling | Turning |
| **Machine** | Röders Tech RFM760 | Spinner U5-630 | ROMI E280 |
| **Workpiece material** | Inconel 718 | Aluminium / PLA | AISI H13 steel |
| **Data type** | Raw time series | Raw time series | Tabular (aggregated) |
| **Sampling frequency** | **50,000 Hz** | **500 Hz** | N/A |
| **Wear label** |  VB continuous |  None |  VBB 3 discrete levels |
| **Labeled samples** | ~945 passes | None | 612 rows |
| **Spectral richness** |  Very high |  Medium |  None |
| **Year** | 2010 | 2025 | 2022 |

---

## 2. Dataset 1 — PHM 2010 Challenge

### 2.1 Origin & Context

- **Source:** PHM Society Conference Data Challenge 2010
- **Organizer:** PHM Society (Prognostics and Health Management)
- **DOI:** `10.21227/jdxd-yy51` (IEEE DataPort)
- **License:** CC0-1.0 (public domain)
- **Academic standing:** World benchmark for milling tool RUL prediction — cited in 500+ publications
- **Download URLs:** see Section 1

### 2.2 Experimental Setup

| Parameter | Value |
|---|---|
| **Machine** | Röders Tech RFM760 (high-speed CNC milling center, Germany) |
| **Milling strategy** | Climb milling (down milling) |
| **Workpiece material** | Inconel 718 (nickel-chromium superalloy, aerospace grade) |
| **Tool type** | 3-flute ball-nose tungsten carbide end mill |
| **Spindle speed** | 10,400 RPM |
| **Feed rate** | 1,555 mm/min |
| **Radial depth of cut (Y)** | 0.125 mm |
| **Axial depth of cut (Z)** | 0.200 mm |
| **Pass length** | 108 mm (X direction) |
| **Cooling** | None (dry milling) |

**Why Inconel 718?** It is notoriously difficult to machine due to high strength at elevated
temperatures, low thermal conductivity, and tendency to work-harden — properties that
accelerate tool wear and make it an ideal study material.

### 2.3 Data Acquisition

| Parameter | Value |
|---|---|
| **Acquisition card** | PCI-1200 |
| **Sampling frequency** | **50,000 Hz (50 kHz) per channel** |
| **Duration per pass** | ~4 seconds (~200,000 data points) |
| **Measurement type** | Synchronous multi-channel |

**Installed sensors:**

| Channel | Sensor type | Measured signal | Unit |
|---|---|---|---|
| `force_x` | Kistler 3-component dynamometer | Cutting force — X axis | N |
| `force_y` | Kistler 3-component dynamometer | Cutting force — Y axis | N |
| `force_z` | Kistler 3-component dynamometer | Cutting force — Z axis | N |
| `vib_x` | Piezoelectric accelerometer | Vibration — X axis | g |
| `vib_y` | Piezoelectric accelerometer | Vibration — Y axis | g |
| `vib_z` | Piezoelectric accelerometer | Vibration — Z axis | g |
| `ae` | Acoustic emission sensor | High-frequency stress waves | V |

**Key characteristic frequencies:**
- Spindle rotation: 10,400 / 60 = **173.3 Hz**
- Tooth passing frequency (3 flutes): 173.3 × 3 = **520 Hz**
- Harmonics: 1,040 Hz, 1,560 Hz, 2,080 Hz, ...
- Nyquist limit at 50 kHz → detectable up to **25,000 Hz**

### 2.4 Wear Measurement

- **Instrument:** Leica MZ12 optical microscope
- **Method:** **Offline** — machine stops after each pass, tool removed and measured
- **Measured quantity:** Flank wear width VB (mm) on each of the 3 cutting edges
- **Unit in files:** 10⁻³ mm (micrometers per flute)
- **End-of-life threshold (ISO standard):** VB = 0.3 mm

### 2.5 File Structure

```
dataset_4/
├── c1/                        ← Tool #1 (labeled )
│   ├── c_1_001.csv            ← Pass 1  [7 columns × ~200,000 rows]
│   ├── c_1_002.csv
│   └── c_1_315.csv            ← Pass 315
├── c2/, c3/, c5/              ← Tools #2, 3, 5 — NO wear labels 
├── c4/                        ← Tool #4 (labeled )
├── c6/                        ← Tool #6 (labeled )
└── wear/
    ├── c1_wear.csv            ← VB after each pass for C1
    ├── c4_wear.csv
    └── c6_wear.csv
```

CSV signal format (7 columns, no header):
`force_x, force_y, force_z, vib_x, vib_y, vib_z, ae`

Wear file format:
`pass, VB_flute1, VB_flute2, VB_flute3`

### 2.6 Key Statistics

| Metric | Value |
|---|---|
| Labeled tools | C1, C4, C6 |
| Total labeled passes | ~945 |
| VB range | 0.000 → ~0.35 mm |
| Classic end-of-life | VB = 0.300 mm |
| Total raw data size | ~50 GB |

### 2.7 Known Limitations

- Single machining condition → limited generalization to other parameters
- Inconel 718 is aerospace-specific, uncommon in general industry
- Only 3 labeled tools → small dataset for deep learning
- Offline wear measurement → no real-time continuity
- Fixed cutting parameters → cannot study parametric effects

---

## 3. Dataset 2 — CNC Machining Energy (Brillinger)

### 3.1 Origin & Context

- **Source:** Pro2Future GmbH + TU Graz, Institute for Production Engineering (Austria)
- **Publication:** *Data in Brief*, Elsevier, 2025 — DOI `10.1016/j.dib.2025.111814`
- **Data DOI:** `10.17632/gtvvwmz7r7.2` (Mendeley Data)
- **License:** CC BY-NC 4.0
- **Funding:** Austrian COMET Programme (FFG 911655) + REDUCE project (FFG 925795)
- **Download URLs:** see Section 1

### 3.2 Experimental Setup

| Parameter | Value |
|---|---|
| **Machine** | Spinner U5-630 (5-axis CNC milling center) |
| **Controller** | Siemens Sinumerik 840D sl v4.8 NCU |
| **Strategy** | Multi-axis milling (free-form surfaces) |
| **Workpiece materials** | AlCuMgPb (aluminium 3.1645) and PLA (26100-51-6) |
| **Raw material dimensions** | 125.3 × 19.34 × 14.52 mm (identical for Al and PLA) |
| **Geometries** | 4 distinct CAD geometries (simple + free-form 2D/3D) |
| **Cooling** | None (dry machining) |
| **CAD/CAM software** | Siemens NX |

Tools used: Ball mill Ø4mm, center drills, chamfer mills, end mills Ø3/6/10/16mm, reamer Ø5mm, twist drills Ø2.8/4.7mm.

### 3.3 Data Acquisition

| Parameter | Value |
|---|---|
| **Edge device** | Siemens Simatic IPC227E |
| **Sampling interval** | **2 ms → 500 Hz per channel** |
| **Protocol** | OPC-UA (open standard, via open62541) |
| **Format** | JSON files (auto-split when too large) |

Recorded signals per axis: commanded/actual position, torque, load, encoder positions,
commanded/actual speed, current, power.

Axes: X, Y, Z, B (fixed=0), Spindle (S), C (fixed=0), Tool changer (T).

### 3.4 File Structure

```
Mendeley_Dataset/
├── CAD_Files/                 ← .STL and .STP for 4 geometries
├── Images/                    ← JPEG photos of machined parts
├── NC_Codes/                  ← G-code programs (.MPF)
├── Tool_Lists/                ← Tool specs (.XLSX)
├── Raw_Datasets/
│   ├── Geometry1_Alu/         ← JSON — aluminium geometry 1
│   ├── Geometry1_PLA/         ← JSON — plastic geometry 1
│   └── ... (8 subfolders total)
├── Pre_Processed_Datasets/    ← Preprocessing scripts (.IPYNB)
└── CNC_Machining_Data_Repository.xlsx
```

### 3.5 Known Limitations

- **No tool wear label** — designed for energy analysis only
- 4 geometries, 1 machine, fixed cutting parameters
- No cooling → non-representative of typical industrial conditions
- Pre-processed data for 2 experiments are missing

---

## 4. Dataset 3 — CNC Turning Roughness & Tool Wear (adorigueto)

### 4.1 Origin & Context

- **Source:** ITA — Instituto Tecnológico de Aeronáutica, Brazil (CCM lab)
- **Author:** Canal A.D., Borille A.V. (Master's dissertation, 2022)
- **Kaggle DOI:** `10.34740/kaggle/ds/2205074`
- **License:** Open (Kaggle)
- **Download URLs:** see Section 1

### 4.2 Experimental Setup

| Parameter | Value |
|---|---|
| **Machine** | ROMI E280 CNC turning center (18.5 kW, 4,000 RPM max) |
| **Process** | Cylindrical turning |
| **Workpiece material** | AISI H13 tool steel (200 HV) |
| **Insert** | Sandvik Coromant ISO TNMG 16 04 04-PF 4425 |
| **Holder** | ISO MTJNL 2020K 16M1 |
| **Cutting fluid** | Blaser Swisslube Vasco 7000 + water (8%), pH ≈ 8 |

### 4.3 Data Acquisition

>  **Tabular data only — no raw time-series signals.**
> Forces are mean values averaged per pass by Kistler Dynoware software.

| Instrument | Model | Measured |
|---|---|---|
| Dynamometer | Kistler 9265B | Triaxial forces (mean/pass) |
| Roughness tester | Mitutoyo Surftest SJ-210 | Ra, Rz (6 spots/pass) |
| Wear microscope | Dino-Lite AM4113ZT | VBB (mm) |

### 4.4 Experimental Design

**Exp1.csv — New tool:** 324 samples, variable Vc/f/ap, VBB ≈ 0 mm
**Exp2.csv — Variable wear:** 288 samples, fixed Vc=350 m/min, VBB ∈ {0.0, 0.1, 0.3} mm
**Prep.csv:** Tool preparation phase (not a planned experiment)

### 4.5 Variables

| Variable | Type | Unit | Description |
|---|---|---|---|
| `Vc` | Input | m/min | Cutting speed |
| `f` | Input | mm/rev | Feed rate |
| `ap` | Input | mm | Depth of cut |
| `Fx`, `Fy`, `Fz` | Input | N | Mean cutting forces |
| `TCond` | Input (Exp2) | — | Tool condition: 0=new, 1=mid, 2=worn |
| `Ra` | **Target** | µm | Arithmetic mean roughness |
| `VBB` | **Target** | mm | Flank wear width |

### 4.6 Known Limitations

- Only 3 discrete wear levels (no continuous wear curve)
- Artificially worn tools (not natural progressive wear)
- No time-series → frequency analysis impossible
- 612 rows total — small dataset
- Single material, single machine

---

## 5. Cross-Dataset Comparison

### 5.1 Feature Extractability

| Feature | PHM 2010 | CNC Machining | CNC Turning |
|---|---|---|---|
| Statistical (mean, std, RMS, kurtosis...) | ✅ | ✅ | ✅ already |
| Frequency bands (FFT) | ✅✅ 50kHz | ✅ 500Hz | ❌ |
| Welch PSD / spectrogram (STFT) | ✅✅ | ✅ | ❌ |
| Energy per operation segment | ❌ | ✅✅ | ❌ |
| Cutting parameters | ❌ fixed | Partial (NC) | ✅✅ |

### 5.2 ML Strategy

| Dataset | Best model | Validation |
|---|---|---|
| PHM 2010 | Random Forest → LSTM / CNN-1D | Leave-One-Tool-Out |
| CNC Machining | Random Forest (energy) | Leave-One-Geometry-Out |
| CNC Turning | RF / XGBoost | Stratified K-Fold |

---

## 6. Usage Recommendations

**Priority 1 — PHM 2010:** Main dataset for wear prediction. Use C1+C4 train, C6 test.

**Priority 2 — CNC Turning:** Complementary. Models parametric effects on wear.

**Supporting — CNC Machining Energy:** No wear label. Useful for energy feature validation.

Suggested pipeline:
```
1. Baseline: PHM 2010 + Random Forest (LOTO)
2. Improve: add FFT features → CNN-1D / LSTM
3. Validate: test on CNC Turning dataset
4. Explore: CNC Machining Energy as auxiliary signal
```

---

## 7. Analysis Notebooks Index

| Notebook | Dataset | Key analyses |
|---|---|---|
| `analysis_PHM2010.ipynb` | PHM 2010 | Signal EDA, FFT, wear evolution, RF + LOTO |
| `analysis_CNC_Machining_Energy.ipynb` | Brillinger 2025 | JSON loading, power profiles, energy features |
| `analysis_CNC_Turning.ipynb` | adorigueto 2022 | EDA, wear effects, Ra/VBB prediction |
| `comparison_datasets.ipynb` | All 3 | Radar, bar charts, feature comparison, conclusions |

---

*Documentation for the CNC tool wear prediction project — April 2026*

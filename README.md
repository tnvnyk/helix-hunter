# helix-hunter

````markdown
# 🧬 DNA Pattern Matching and Alignment Tool

## 🚨 The Challenge & Motivation

DNA sequence analysis is foundational to bioinformatics, yet exact pattern matching in massive biological data can be computationally intensive. Researchers often need to identify specific gene sequences within larger genomes to study genetic markers, mutations, or conserved regions.

This project addresses that challenge by combining classic computer science algorithms with biological data processing — providing a fast, benchmarked, and extensible tool to perform DNA pattern matching.

---

## ⚙️ What This Project Does

This tool allows users to:

- Search for a specific DNA pattern in a `.fasta` file using:
  -  **Naive Brute Force**
  -  **Knuth-Morris-Pratt (KMP)**
  -  **Rabin-Karp (Rolling Hash)**
- Benchmark these algorithms by measuring:
  - Execution time
  - Number of matches found
- Optionally perform **local alignment** to extend matches, inspired by the **BLAST** (Basic Local Alignment Search Tool) methodology.

---

## ▶ Running the Project

### 1 Command Line Interface (CLI)

Run from terminal with the following options:

```bash
# Basic usage
python main.py --file <your_fasta_file.fasta> --pattern <ATCG>

# Run all algorithms with plots
python main.py --file test_data/sample.fasta --pattern ATCG --algorithm all --plot

# Enable local alignment (BLAST-inspired)
python main.py --file test_data/sample.fasta --pattern ATCG --extend --window 8
````

#Available algorithms:

* `naive`
* `kmp`
* `rk`
* `all`

---

### 2️ Gradio Web UI

Launch the UI by simply running:

```bash
python main.py
```

This opens an interactive browser-based interface to upload FASTA files, enter a pattern, and select the matching algorithm.

---





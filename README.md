# Molecular signal determines the sensitivity of phylogenetic node height estimates to demographic assumptions in Bayesian frameworks

Time-calibrated phylogenies are widely used to reconstruct pathogen evolution and transmission, but in a Bayesian framework node height estimates depend on both molecular information and assumptions about population dynamics. In this study, we investigate when demographic assumptions influence estimates of node heights in dated phylogenies.
We derive analytical posterior distributions for the time to the most recent common ancestor of two sequences under constant population size, exponential growth, and bottlenecks. We show that demographic assumptions can have a large impact on node height estimates, while the influence decreases with increasing molecular signal and population sizes.
We then simulate phylogenies and sequence data and re-estimate node times on fixed topologies using constant-size and skyline coalescent priors. Demographic assumptions have little effect when molecular signal is high, but increasingly affect deep node height estimates as signal declines. The skyline prior reduces these errors under strong population changes.
Finally, we analyse two case studies. For the initial SARS-CoV-2 Alpha variant wave in the UK, a misspecified constant-size prior substantially shifts the estimated root towards the past, whereas a skyline prior recovers more plausible timing. For \textit{Bordetella pertussis} in France, where transmission was halted during the COVID-19 pandemic, a constant-size prior similarly pushes internal node estimates towards older dates.
Overall, our results demonstrate that demographic misspecification can bias the dating of phylogenies, particularly when molecular signal is limited or population size undergoes rapid changes to low levels.

## Part 3: Data Analysis of a Pertussis and a SARS-CoV-2 example

This repository holds the empirical part of the study. For each of two data sets we run BEAST twice on the same alignment: once with a **skygrid** prior and once with a **constant-size coalescent** prior. We then compare the two posteriors in terms of:

1. inferred population size trajectories and the root height (tMRCA),
2. summary (MCC) trees and their topologies (tanglegrams, Robinson-Foulds distances), and
3. node heights, measured as the distance of every pair of tips to their pairwise MRCA, summarised over 1000 posterior trees.

### Data sets

| Data set | Folder | Period | Expected population dynamics |
|---|---|---|---|
| *Bordetella pertussis*, France | `data/pertussis_france/` | 2010–2024 | Transmission halted during the COVID-19 pandemic (bottleneck from 2020) |
| SARS-CoV-2 Alpha variant (B.1.1.7), UK | `data/sarscov/` | 2020 | Rapid exponential growth during the initial wave |

The pertussis isolates and metadata come from the supplementary information of Bouchez et al. (2025) [[1]](#references), restricted to French isolates sampled from 2010 onwards. For SARS-CoV-2, we use data described in Hill et al. (2022) [[2]](#references) and compare the inferred tMRCA with their estimate.

Large files such as the alignments, BEAST logs and posterior tree files are not tracked in git (see `.gitignore`).

The exact sampling dates of the pertussis isolates are confidential and not shared. The repository includes `data/pertussis_france/metadata.csv` (isolate, year, country and genotypes) but not `full_metadata.csv`, which also contains the exact sampling date as a decimal year. The evaluation notebooks read `full_metadata.csv`, so the parts that need exact sampling dates cannot be re-run without it.

### Repository structure

```
├── environment.yaml                  # conda environment (Python packages, snp-sites)
├── preprocess_pertussis_data.ipynb   # metadata, alignment preparation and BEAST runs (pertussis)
├── evaluate_pertussis_data.ipynb     # evaluation of the pertussis BEAST runs
├── evaluate_sarscov2_data.ipynb      # evaluation of the SARS-CoV-2 BEAST runs
├── evaluation_helpers.py             # summary, plotting and tree comparison functions used by the notebooks
├── scripts/                          # preprocessing and cluster (SLURM) scripts
├── data/
│   ├── pertussis_france/             # metadata, gubbins output, BEAST runs, tip/topology distances
│   └── sarscov/                      # metadata, BEAST runs, PhyCLIP clusters, tip/topology distances
└── figures/                          # figures produced by the evaluation notebooks
```

### Workflow

#### 1. Preprocessing (pertussis only, `preprocess_pertussis_data.ipynb`)

1. **Metadata:** filter the isolates to France and 2010 or later, then match each isolate to its exact sampling date and convert the date to a decimal year (`data/pertussis_france/full_metadata.csv`).
2. **Alignment:** subsample the whole-genome alignment to the selected isolates (`scripts/subsample_fasta.py`, 1133 of 1588 sequences kept).
3. **Recombination:** remove recombinant regions with Gubbins (`scripts/run_gubbins.sh`), then convert the Gubbins tree to NEXUS so it can be used as a starting tree in BEAUti.
4. **SNPs:** keep only polymorphic sites with `snp-sites` (3288 SNPs). Count the invariant bases with `scripts/count_invariant_bases.py` so they can be added to the BEAST XML as constant-site counts.

#### 2. BEAST inference

For pertussis, each prior was run with three random seeds (`scripts/run_beast_skygrid_pertussis.sh`, `scripts/run_beast_constcoal_pertussis.sh`). Each run was split into consecutive 120-hour cluster jobs that resume from the last checkpoint. The runs were then combined, and summary trees were computed with TreeAnnotator, keeping node heights (`scripts/run_treeannotator.sh --heights keep`).

The SARS-CoV-2 BEAST XMLs and combined runs are in `data/sarscov/beast_skygrid/` and `data/sarscov/beast_constcoal/`.

#### 3. Posterior tree comparisons (cluster)

Both comparisons run on 1000 trees subsampled from each posterior:

- **Topology:** `scripts/run_topology_distances.sh` runs `scripts/compute_topology_distances.py`. It computes Robinson-Foulds distances within the skygrid trees, within the constant-size coalescent trees, and between the two priors, and writes `topology_distances_1000.csv`.
- **Node heights:** `scripts/run_coalescent_distances.sh` runs `scripts/compute_coalescent_distances.py`. For every pair of tips and every posterior tree, it computes the distance from each tip to the pair's MRCA. It reports the median and 95% HPD per prior, plus the absolute and relative difference (constant-size − skygrid), and writes `coalescent_tip_distances_1000.tsv`.

Both scripts take the skygrid trees, the constant-size coalescent trees and an output path as arguments. Without arguments they default to the pertussis files.

#### 4. Evaluation (`evaluate_pertussis_data.ipynb`, `evaluate_sarscov2_data.ipynb`)

Each evaluation notebook:

- summarises root height and tree length (median, 95% HPD) for both priors
- plots the skygrid trajectory against the constant population size, together with both tMRCAs (`*_popsize*.pdf`). For SARS-CoV-2, it also fits an exponential growth rate to the skygrid trajectory and adds the tMRCA from Hill et al.
- plots the summary time trees (pertussis, coloured by *ptxP* × *fim3* genotype) and tanglegrams of the skygrid and constant-size coalescent summary trees. Pertussis tanglegrams are coloured by genotype; SARS-CoV-2 tanglegrams are coloured by sampling date or by PhyCLIP cluster. For pertussis, a tanglegram of two skygrid posterior samples serves as a reference for how much topology varies within one posterior
- plots the distributions of Robinson-Foulds distances within and between priors (`rf_distributions*.pdf`)
- compares tip-to-MRCA distances under the two priors, as a scatter plot with a running median, a zoom panel and a panel of relative or absolute differences (`*pairwise_tip_distances*.pdf`). For pertussis, this is repeated for pairs where both tips were sampled after the 2020 bottleneck (`bottleneck_*`). For SARS-CoV-2, points are coloured by whether both tips were sampled after mid-November 2020.

### Main figures

| Figure | Content |
|---|---|
| `pertussis_popsize*.pdf`, `sarscov_popsize*.pdf` | Skygrid vs. constant population size with posterior samples, 95% HPD and tMRCAs |
| `pertussis_skygrid_timetree.pdf`, `pertussis_constcoal_timetree.pdf` | Pertussis MCC time trees coloured by *ptxP* × *fim3* genotype |
| `tanglegram*.pdf`, `twiceskygrid_tanglegram.pdf`, `sarscov_tanglegram*.pdf` | Skygrid (left) vs. constant-size coalescent (right) summary trees |
| `rf_distributions*.pdf` | Robinson-Foulds distances within and between priors |
| `pairwise_tip_distances*.pdf`, `bottleneck_pairwise_tip_distances*.pdf` | Tip-to-pairwise-MRCA distances: constant-size coalescent vs. skygrid |

### Requirements

The analyses use Python with `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn` and `dendropy`. External tools: BEAST v1.10.4 with BEAGLE, TreeAnnotator, Gubbins and snp-sites. The cluster scripts assume a SLURM environment. The environment is specified in `environment.yaml`.

### References

1. Bouchez V, Lefrancq N, Toubiana J, Rodrigues C, Brisse S. Microbes know no borders: importation of macrolide-resistant *Bordetella pertussis* into France in 2024. *Clinical Microbiology and Infection* 31(10):1737–1739 (2025). [doi:10.1016/j.cmi.2025.06.025](https://doi.org/10.1016/j.cmi.2025.06.025)
2. Hill V, Du Plessis L, Peacock TP, *et al.* The origins and molecular evolution of SARS-CoV-2 lineage B.1.1.7 in the UK. *Virus Evolution* 8(2):veac080 (2022). [doi:10.1093/ve/veac080](https://doi.org/10.1093/ve/veac080)

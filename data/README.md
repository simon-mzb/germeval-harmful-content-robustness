# data/ — GermEval 2025 and 2026

The datasets are **not redistributed** with this repository. They are published by the
shared-task organisers in the GitHub repository
[`Communication-Forensics-Lab/harmful-content-detection`](https://github.com/Communication-Forensics-Lab/harmful-content-detection)
under the GPL-3.0; the competitions ran on Codabench (2025: competition 4963,
2026: competition 14006).

## Download

```bash
git clone https://github.com/Communication-Forensics-Lab/harmful-content-detection.git /tmp/gehcd
git -C /tmp/gehcd checkout 3f9e11e        # the state every result here was produced from
mkdir -p data/codabench
cp -r /tmp/gehcd/GermEval2025 /tmp/gehcd/GermEval2026 data/codabench/
rm -rf /tmp/gehcd
```

Commit `3f9e11e` (3 August 2026) is the upstream state the 21 data files were checked
against; the data files themselves last changed on 25 May 2026.

## Expected layout

```
data/codabench/
├── GermEval2025/data/{c2a,dbo,vio}/{c2a,dbo,vio}_{train,trial,test}.csv
└── GermEval2026/data/{c2a,dbo,vio,def}/..._{train,trial,test}[_26].csv
```

Files are `;`-separated with the columns `id;description[;<label>]`. Test files of both
editions ship without gold labels, so every evaluation in this repository runs on the
training data (cross-validation on 2025, a held-out split on 2026).

## Integrity check

`results/data_manifest.json` records the SHA-256, size and line count of all 21 CSV
files. `src.data_loading.load_split` verifies the whole tree against it on first use and
raises on any mismatch, so a partial or changed download cannot silently produce numbers.
To check by hand:

```bash
uv run python -m src.data_manifest --verify
```

## Task descriptions (E5 sample only)

`src.e5_label_sample` writes the coding instructions for the manual error analysis and
quotes the organisers' class definitions from `data/codabench/elaborated_info_2025_tasks.txt`,
a plain-text copy of the 2025 task description on the Codabench competition page, with
one `# Subtask <n>: <name>` heading per subtask. It is not part of the upstream repository
and is not shipped here, because it is the organisers' text and quotes items of the
dataset.

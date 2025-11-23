# **unduliner**

### *Predicting methylation-altering single nucleotide variants (SNVs) from Nanopore sequencing*

`unduliner` is a Python package designed to predict whether a somatic single-nucleotide variant is associated with local methylation change, 
using Nanopore long-read sequencing and a pre-trained deep learning model.

The tool takes as input:

* One BAM file (Nanopore reads with MM/ML tags)
* One VCF file (somatic SNVs)
* A trained model (`.pth`, included in the `model/` directory)

It extracts genomic regions around each SNV, evaluates methylation differences using a statistical and deep learning pipeline, and predicts whether 
the variant is likely to cause a methylation shift.

---

## **Features**

* Predicts SNV-associated methylation alterations
* Works directly on Nanopore MM/ML–tagged BAMs
* User-configurable methylation thresholds
* Outputs an interpretable table including DMR-like regions and methylation deltas

---

## **Installation**

### **Clone the repository**

```bash
git clone https://github.com/Chisomgold/unduliner.git
cd unduliner
```

### **Create environment**

```bash
conda env create -f unduliner.yml
conda activate unduliner
pip install -e $PWD
unduliner -h
```



---

## **Usage**

### **Required arguments**

| Argument  | Description                                              |
| --------- | -------------------------------------------------------- |
| `--bam`   | Nanopore BAM file with MM/ML tags                        |
| `--vcf`   | VCF file containing SNVs                                 |
| `--model` | Path to trained `.pth` model (default model in `model/`) |

### **Some optional arguments**

| Argument          | Default                | Description                                       |
| ----------------- | ---------------------- | ------------------------------------------------- |
| `--chromosome`    | -                      | Chromosome of interest (e.g., chr10)              |
| `--region`        | -                      | Region of interest (e.g. chr10:3000-4000)         |
| `--mincov`        | 3                      | Minimum read coverage per allele                  |
| `--mincpgs`       | 3                      | Minimum CpGs required within region               |
| `--cpgdist`       | 50                    | Maximum distance allowed between consecutive CpGs |
| `--meth_cutoff`   | 0.8                    | Probability threshold for calling methylated      |
| `--unmeth_cutoff` | 0.2                    | Probability threshold for calling unmethylated    |

---

## **Example command**

```bash
unduliner -b testdata/chr17_2M_225M.bam -v testdata/chr17atcc.vcf.gz --model model/pretrained80.pth --mincov 5 --mincpgs 10 
```

---

# **Output**

`unduliner` generates a tab-delimited table summarising methylation effects per variant.

### **Example Output Table**

| Chr   | Start     | End       | Ref | Alt | Reads_with_Ref | Reads_with_Alt | Prediction | Top 5 DMHs                  | Meth-prop-diffs      |
| ----  | --------- | --------- | --- | --- | -------------  | -------------- | ---------- | --------------------------- | -------------------- |
| chr17 | 21303243  | 21303244  | C   | T   | 24             | 26             | Positive   | 21297142-21297286=3.365e-09 | -0.387               |
| chr17 | 22285281  | 22285282  | A   | G   | 8              | 20             | Positive   | 22279656-22279851=1.070e-06 | 0.397                |
| chr17 | 20000171  | 20000172  | A   | C   | 8              | 16             | Negative   | NA                          | NA                   |
| chr17 | 20002511  | 20002512  | A   | G   | 24             | 5              | Amb        | NA                          | NA                   |
| chr17 | 20002617  | 20002618  | C   | T   | 12             | 16             | Amb        | NA                          | NA                   |


### Column descriptions

* **Prediction:** “Positive” = variant likely associated with nearby methylation change, "Amb" = unsure (e.g., insufficient reads)
* **Top 5 DMHs:** <= 5 Genomic intervals with smallest adjusted p-values
* **Meth-prop-diffs:** Δ(methylation) between ref-allele reads and alt-allele reads, where a negative value means reduced methylation on the variant reads compared to the reference.
* **Start/End:** The SNV genomic coordinate (End = variant position)

---


## **Citing unduliner**

If you use **unduliner** in your research, please cite the repository:


A manuscript is in preparation.

---

## **License**

MIT License

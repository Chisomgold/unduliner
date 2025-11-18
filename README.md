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
unduliner -b tumor_nanopore.bam -v somatic_calls.vcf.gz --model model/pretrained96.pth --mincov 5 --mincpgs 10 
```

---

# **Output**

`unduliner` generates a tab-delimited table summarising methylation effects per variant.

### **Example Output Table**

| Chr  | Start     | End       | Ref | Alt | Reads_with_Ref | Reads_with_Alt | Prediction | Top 5 DMHs                                                                                                                                              | Meth-prop-diffs                    |
| ---- | --------- | --------- | --- | --- | -------------- | -------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| chr1 | 200042105 | 200042106 | T   | C   | 15             | 10             | Positive   | 200042002-200043178=1.379e-184;200040638-200041704=2.124e-150;200029088-200029544=1.236e-81;200038137-200038802=2.986e-63;200040203-200040429=4.997e-23 | -0.833;-0.803;-0.946;-0.806;-0.810 |
| chr1 | 153256878 | 153256879 | A   | G   | 16             | 16             | Positive   | 153261276-153261805=3.930e-165;153261051-153261163=7.157e-53                                                                                            | -1.000;-1.000                      |
| chr1 | 18632328  | 18632329  | C   | T   | 10             | 16             | Positive   | 18631507-18631873=6.066e-111;18632982-18633255=2.008e-68;18631105-18631312=3.120e-64;18630333-18630917=3.723e-62;18636257-18636591=6.214e-60            | -0.896;-0.888;-0.915;-0.932;-0.908 |
| chr1 | 46400100  | 46400101  | G   | T   | 11             | 8              | Positive   | 46394134-46394606=8.889e-58                                                                                                                             | 0.658                              |
| chr1 | 41189666  | 41189667  | A   | T   | 11             | 17             | Negative   | 41185845-41185915=1.745e-01                                                                                                                             | -0.190                             |
| chr1 | 106015317 | 106015318 | C   | T   | 4              | 16             | Negative   | NA                                                                                                                                                      | NA                                 |
| chr1 | 1077886   | 1077887   | C   | T   | 5              | 12             | Negative   | NA                                                                                                                                                      | NA                                 |

### Column descriptions

* **Prediction:** “Positive” = variant likely associated with nearby methylation change
* **Top 5 DMHs:** Genomic intervals with smallest adjusted p-values
* **Meth-prop-diffs:** Δ(methylation) between ref-allele reads and alt-allele reads, where a negative value means reduced methylation on the variant reads compared to the reference.
* **Start/End:** The SNV genomic coordinate (End = variant position)

---


## **Citing unduliner**

If you use **unduliner** in your research, please cite the repository:


A manuscript is in preparation.

---

## **License**

MIT License

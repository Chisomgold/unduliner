# **unduliner**

### *Predicting methylation-altering single nucleotide variants (SNVs) and structural variants (SVs) from nanopore sequencing*

`unduliner` is a Python package designed to predict whether a somatic variant is associated with local methylation change, 
using nanopore long-read sequencing and a pre-trained deep learning model.

The tool takes as input:

* One BAM file (Nanopore reads with MM/ML tags)
* One VCF file
* A trained model (`.pth`, included in the `model/` directory and added by default to options)

It extracts genomic regions around each variants, evaluates methylation differences using a statistical and deep learning pipeline, 
and predicts whether the variant is likely to cause a methylation shift.

---

## **Features**

* Predicts SNV- and SV-associated methylation alterations
* Works directly on Nanopore MM/ML–tagged BAMs
* User-configurable methylation thresholds
* Outputs an interpretable table including DMRs and methylation deltas
* Supports GTF and regulatory elements annotation of identified regions

---

## **Installation**

### `pip install unduliner` in an environment with samtools installed.

```bash
conda create -n unduliner python=3.10
conda activate unduliner
conda install -c bioconda samtools
pip install unduliner
unduliner
```

### **Alternatively, clone the repository**

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

### **Some optional arguments**

| Argument          | Default                | Description                                                            |
| ----------------- | ---------------------- | --------------------------------------------------------------------   | 
| `--chromosome`    | -                      | Chromosome of interest (e.g., chr10)                                   |
| `--region`        | -                      | Region of interest (e.g. chr10:3000-4000)                              |
| `--mincov`        | 3                      | Minimum read coverage per allele                                       |
| `--mincpgs`       | 3                      | Minimum CpGs required within region                                    |
| `--cpgdist`       | 50                     | Maximum distance allowed between consecutive CpGs                      |
| `--meth_cutoff`   | 0.8                    | Probability threshold for calling methylated                           |
| `--unmeth_cutoff` | 0.2                    | Probability threshold for calling unmethylated                         |
| `--tmp`           | working dir            | Dir that supports read/write of many files                             |
| `--gtf`           | -                      | Tabix-indexed GTF file to annotated diff methylated regions            |
| `--cre`           | -                      | BED-style file (4 cols: chr, start, end, feature name) for annotation  |
| `--sv`            | -                      | Activates functions for SVs - INS,DEL,BND,DUP,INV                      |
---

## **Example command**

```bash
unduliner -b testdata/chr17_2M_225M.bam -v testdata/chr17atcc.vcf.gz 

#optionally add --gtf path/to/sorted.gtf.gz --cre path/to/promoters.bed --cre path/to/repeatsWG.bed
```

### for SV
```bash
unduliner -b sample.bam -v SV.vcf.gz --sv
```


---

# **Output**

`unduliner` generates a tab-delimited table summarising methylation effects per variant.

### **Minimal Output Table**

| Chr   | Start     | End       | Ref | Alt | Reads_with_Ref | Reads_with_Alt | Prediction | Top 5 DMHs                  | Meth-prop-diffs  |
| ----  | --------- | --------- | --- | --- | -------------  | -------------- | ---------- | --------------------------- | ---------------- |
| chr17 | 21552714  | 21552715  | C   | T   | 26             | 6              | Positive   | 21551211-21551853=2.854e-56 | 0.642            |
| chr17 | 20194148  | 20194149  | A   | G   | 8              | 25             | Negative   | 20174773-20174843=5.040e-02 | 0.376            |  
| chr17 | 21276148  | 21276149  | C   | A   | 8              | 12             | Amb        | 21275234-21276360=2.852e-04 | 0.180            |



### Column descriptions

* **Prediction:** “Positive” = variant likely associated with nearby methylation change, "Amb" = unsure (e.g., insufficient reads)
* **Top 5 DMHs:** <= 5 Genomic intervals with smallest adjusted p-values
* **Meth-prop-diffs:** Δ(methylation) between ref-allele reads and alt-allele reads, where a negative value means reduced 
methylation on the variant reads compared to the reference.
* **Start/End:** The SNV genomic coordinate (End = variant position)
* Extra columns for GTF and CREs by user-request. The `--cre` option can be used multiple times; files should be tab/space-delimited.
Only the first 4 columns will be used with chr, start, end, as the first 3 columns respectively and the fourth column is a 
genomic feature relevant to that region like promoter, enhancer, repeats, etc. 

---


## **Citing unduliner**

If you use **unduliner** in your research, please cite the repository:


A manuscript is in preparation.

---

## **License**

MIT License

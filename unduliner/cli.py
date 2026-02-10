#!/usr/bin/env python

import os
import sys
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
from torchvision.datasets import ImageFolder
import warnings
warnings.filterwarnings("ignore")
import pysam
pysam.set_verbosity(0)
import argparse
import csv
import subprocess
from uuid import uuid4
import pandas as pd
from collections import defaultdict as dd
from bx.intervals.intersection import Intersecter, Interval
from scipy.stats import ttest_ind, fisher_exact
from statsmodels.stats import multitest
from statsmodels.stats.proportion import proportions_ztest
from sklearn.mixture import GaussianMixture
from importlib.resources import files

import logging
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_model_path():
    return files("unduliner.model").joinpath("pretrained80.pth")

MODEL_PATH = get_model_path()

# Define architecture
class Net(nn.Module):
    def __init__(self, input_size, num_layers):
        super(Net, self).__init__()
        layers = []
        # initial input and output channels
        c_in = 3
        c_out = 256

        for i in range(num_layers):
            layers.append(self.conv_block(c_in=c_in, c_out=c_out, dropout=0.4 if i == 0 else 0.25, kernel_size=3 if i != 0 else 5, stride=1, padding=1 if i != 0 else 2))
            c_in = c_out
            c_out = c_out // 2
            if (i + 1) % 2 == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(in_channels=c_in, out_channels=2, kernel_size=input_size // (2 ** ((num_layers + 1) // 2)), stride=1, padding=0))
        layers.append(nn.Flatten())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )

        return seq_block

#1: Separate reads by SNV
def separate_reads_by_variant(bam_file, vcf_file, output_dir, args):
    logger.info("reading input files")
    bam = pysam.AlignmentFile(bam_file, 'rb')
    vcf_reader = pysam.VariantFile(vcf_file)

    region_chrom = None
    region_start = None
    region_end = None

    variant_count = 0
    if args.chromosome:
        region_chrom = args.chromosome

    if args.region:
        region_chrom, coords = args.region.split(':')
        region_start, region_end = map(int, coords.split('-'))

    empty_bams = {"alt": [], "ref": []} #for alleles with no supporting reads

    for record in vcf_reader.fetch():
        chrom = record.chrom
        start = record.pos - 1
        end = start + len(record.ref)
        alt = record.alts[0]
        ref = record.ref

        #skip regions not matching user input
        if region_chrom and chrom != region_chrom:
            continue

        if region_start and region_end and (start < region_start or end > region_end):
            continue

        with_variant = pysam.AlignmentFile(f'{output_dir}/{chrom}_{start}_{end}_{ref}_{alt}_with_variant.bam', 'wb', template=bam)
        without_variant = pysam.AlignmentFile(f'{output_dir}/{chrom}_{start}_{end}_{ref}_{alt}_without_variant.bam', 'wb', template=bam)

        reads = bam.fetch(chrom, start, end)
        read_list = list(reads)

        wrote_alt = False
        wrote_ref = False

        for read in read_list:
            if read.query_sequence is None:
                continue
            for positions in read.get_aligned_pairs():
                if positions[0] is None:
                    continue
                if positions[1] == start:
                    if read.query_sequence[positions[0]] == alt:
                        with_variant.write(read)
                        wrote_alt = True
                        logger.debug(f"wrote {read.query_name} to with_variant; alt, ref is {alt}, {ref}; read has {read.query_sequence[positions[0]]}")
                    elif read.query_sequence[positions[0]] == ref:
                        without_variant.write(read)
                        wrote_ref = True
                        logger.debug(f"wrote {read.query_name} to without_variant; alt, ref is {alt}, {ref}; read has {read.query_sequence[positions[0]]}")

        if not wrote_alt or not wrote_ref:
            if not wrote_alt:
                empty_bams["alt"].append((chrom, start, end, ref, alt))

            if not wrote_ref:
                empty_bams["ref"].append((chrom, start, end, ref, alt))

            with_variant.close()
            without_variant.close()
            os.remove(f'{output_dir}/{chrom}_{start}_{end}_{ref}_{alt}_with_variant.bam')
            os.remove(f'{output_dir}/{chrom}_{start}_{end}_{ref}_{alt}_without_variant.bam')
            continue
        else:
            variant_count += 1

        if variant_count % 100 == 0 and variant_count != 0:
            logger.info(f"Processed {variant_count} variants...")

        with_variant.close()
        without_variant.close()

    report_path = "read_parse_summary.txt"

    if empty_bams["alt"] or empty_bams["ref"]:
        with open(report_path, "w") as report:
            report.write("Skipped the following positions because at least one BAM was empty:\n")

            if empty_bams["alt"]:
                report.write("\nNo ALT-supporting reads for:\n")
                for chrom, start, end, ref, alt in empty_bams["alt"]:
                    report.write(f"{chrom}:{start}-{end} REF={ref} ALT={alt}\n")

            if empty_bams["ref"]:
                report.write("\nNo REF-supporting reads for:\n")
                for chrom, start, end, ref, alt in empty_bams["ref"]:
                    report.write(f"{chrom}:{start}-{end} REF={ref} ALT={alt}\n")

    logger.info(f"Processed a total of {variant_count} variants")
    bam.close()
    vcf_reader.close()

#SV H:
def read_supports_INS(read, pos, svlen):
    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op in (0, 2, 3, 7, 8): #consume ref
            ref_pos += length
        elif op == 1:
            if abs(ref_pos - pos) <= 1:
                if abs(length - svlen) <= 10:
                    return True
        elif op in (4, 5):
            pass

    return False

def read_supports_DEL(read, pos, svlen):
    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op == 2:
            if abs(ref_pos - pos) <= 1:
                if abs(length - svlen) <= 10:
                    return True
            else:
                ref_pos += length
        elif op in (0, 3, 7, 8):
            ref_pos += length
        else:
            pass

    return False

def parse_bnd_alt(alt):
    bnd = re.search(r'[\[\]]([^:\[\]]+):(\d+)[\[\]]', alt.replace(" ", ""))
    if not bnd:
        return None, None
    return bnd.group(1), int(bnd.group(2))

def read_supports_BND(read, chrom, pos, alt):
    partner_chrom, partner_pos = parse_bnd_alt(alt)
    if partner_chrom is None:
        return False

    #BND would be at the start/end of the primary read
    if not (abs(read.reference_start  - pos) <= 10 or abs(read.reference_end - pos) <= 10):
        return False

    def norm(c):
        return c if c.startswith("chr") else "chr" + c

    chrom = norm(chrom)
    partner_chrom = norm(partner_chrom)

    primary = (read.reference_name == chrom)

    if read.has_tag("SA"):
        for aln in read.get_tag("SA").split(";"):
            if not aln:
                continue
            c, p = aln.split(",")[0:2]
            c = norm(c)
            if c == partner_chrom and abs(int(p) - partner_pos) <= 10:
                return True

    return False

def refspan(cigar):
    span = 0
    for length, op in re.findall(r'(\d+)([MIDNSHP=X])', cigar):
        length = int(length)
        if op in ("M", "D", "=", "X"):
            span += length
    return span


def read_supports_INV(read, chrom, end):
    if not read.has_tag("SA"):
        return False

    if read.reference_name != chrom:
        return False
    for aln in read.get_tag("SA").split(";"):
        if not aln:
            continue
        fields = aln.split(",")
        sa_chrom, sa_pos, sa_strand, sa_cigar = fields[0], int(fields[1]), fields[2], fields[3]
        ref_span = refspan(sa_cigar)
        if sa_chrom == read.reference_name:
            if abs((sa_pos + ref_span) - end) <= 1:
                # strand switch
                if (sa_strand == "-" and not read.is_reverse) or (sa_strand == "+" and read.is_reverse):
                    return True
    return False

def read_supports_DUP(read, chrom, end):
    if not read.has_tag("SA"):
        return False

    if read.reference_name != chrom:
        return False
    for aln in read.get_tag("SA").split(";"):
        if not aln:
            continue
        fields = aln.split(",")
        sa_chrom, sa_pos, sa_strand, sa_cigar = fields[0], int(fields[1]), fields[2], fields[3]
        ref_span = refspan(sa_cigar)
        if sa_chrom == read.reference_name:
            if abs((sa_pos + ref_span) - end) <= 1:
                if (sa_strand == "-" and read.is_reverse) or (sa_strand == "+" and not read.is_reverse):
                    return True
    return False

#2: Separate reads by SV
def separate_reads_by_sv(bam_file, vcf_file, output_dir, args):
    logger.info("reading input files")
    bam = pysam.AlignmentFile(bam_file, 'rb')
    vcf_reader = pysam.VariantFile(vcf_file)

    sv_count = 0
    empty_bams = {"alt": [], "ref": []}

    region_chrom = None
    region_start = None
    region_end = None

    if args.chromosome:
        region_chrom = args.chromosome

    if args.region:
        region_chrom, coords = args.region.split(':')
        region_start, region_end = map(int, coords.split('-'))

    for record in vcf_reader.fetch():
        svtype = record.info['SVTYPE']
        chrom = record.chrom
        start = record.pos - 1
        ref = record.ref
        if len(ref) > 6:
            ref = str(ref[0]) + str(len(ref)-2) + str(ref[-1])
        alt = record.alts[0]
        end = record.stop
        svlen = abs(record.info.get("SVLEN", end - start))

        if region_chrom and chrom != region_chrom:
            continue

        if region_start and region_end and (start < region_start or end > region_end):
            continue

        with_variant = pysam.AlignmentFile(f'{output_dir}/{chrom}_{start}_{end}_{ref}_{svtype}_with_variant.bam', 'wb', template=bam)
        without_variant = pysam.AlignmentFile(f'{output_dir}/{chrom}_{start}_{end}_{ref}_{svtype}_without_variant.bam', 'wb', template=bam)

        reads = bam.fetch(chrom, start, end)

        wrote_alt = False
        wrote_ref = False

        for read in reads:
            if read.query_sequence is None:
                continue

            if svtype == "INS":
                supports = read_supports_INS(read, start, svlen)

            elif svtype == "DEL":
                supports = read_supports_DEL(read, start, svlen)

            elif svtype == "BND":
                supports = read_supports_BND(read, chrom, start, alt)

            elif svtype == "INV":
                supports = read_supports_INV(read, chrom, end)

            elif svtype == "DUP":
                supports = read_supports_DUP(read, chrom, end)

            else:
                continue

            if supports:
                with_variant.write(read)
                wrote_alt = True
                logger.debug(f"wrote {read.query_name} to with_variant")

            else:
                without_variant.write(read)
                wrote_ref = True
                logger.debug(f"wrote {read.query_name} to without_variant")

        if not wrote_alt or not wrote_ref:
            if not wrote_alt:
                empty_bams["alt"].append((chrom, start, end, svtype))

            if not wrote_ref:
                empty_bams["ref"].append((chrom, start, end, svtype))

            with_variant.close()
            without_variant.close()
            os.remove(f'{output_dir}/{chrom}_{start}_{end}_{ref}_{svtype}_with_variant.bam')
            os.remove(f'{output_dir}/{chrom}_{start}_{end}_{ref}_{svtype}_without_variant.bam')
            continue
        else:
            sv_count += 1

        if sv_count % 100 == 0 and sv_count != 0:
            logger.info(f"Processed {sv_count} variants...")

        with_variant.close()
        without_variant.close()

    report_path = "read_parse_summary.txt"

    if empty_bams["alt"] or empty_bams["ref"]:
        with open(report_path, "w") as report:
            report.write("Skipped the following positions because at least one BAM was empty:\n")

            if empty_bams["alt"]:
                report.write("\nNo ALT-supporting reads for:\n")
                for chrom, start, end, svtype in empty_bams["alt"]:
                    report.write(f"{chrom}:{start}-{end};{svtype}\n")

            if empty_bams["ref"]:
                report.write("\nNo REF-supporting reads for:\n")
                for chrom, start, end, svtype in empty_bams["ref"]:
                    report.write(f"{chrom}:{start}-{end};{svtype}\n")

    logger.info(f"Processed a total of {sv_count} variants")

    bam.close()
    vcf_reader.close()

#H: index BAM files
def index_bam_files(output_dir):
    logger.info("Indexing bam files")
    for bam_file in os.listdir(output_dir):
        if bam_file.endswith(".bam"):
            bam_file_path = os.path.join(output_dir, bam_file)
            sorted_bam = bam_file_path.replace(".bam", "_sorted.bam")

            if not os.path.exists(sorted_bam):
                subprocess.run(["samtools", "sort", "-o", sorted_bam, bam_file_path])
            subprocess.run(["samtools", "index", sorted_bam])

#3. Extract methylation data
def create_methylation_array(bam_path, meth_cutoff=0.8, unmeth_cutoff=0.2, mem=8):
    MAX_ARRAY_BYTES = mem * 1024**3  # 8 GB safety limit
    bam = pysam.AlignmentFile(bam_path, "rb")
    methylation_data = {}

    for read in bam:
        if read.query_sequence is None:
            continue
        read_id = read.query_name
        aligned_positions = {query_pos: ref_pos for query_pos, ref_pos in read.get_aligned_pairs(matches_only=True)}

        if read.modified_bases is None:
            continue
        for key, values in read.modified_bases.items():
            for base_pos, quality in values:
                probability = 0
                if quality / 255.0 > meth_cutoff:
                    probability=1
                if quality / 255.0 < unmeth_cutoff:
                    probability=-1
                if base_pos in aligned_positions:
                    ref_pos = aligned_positions[base_pos]
                    if ref_pos not in methylation_data:
                        methylation_data[ref_pos] = {}
                    methylation_data[ref_pos][read_id] = probability

    genomic_positions = sorted(methylation_data.keys())
    read_ids = list({read_id for pos_data in methylation_data.values() for read_id in pos_data})

    n_rows = len(genomic_positions)
    n_cols = len(read_ids)
    estimated_bytes = n_rows * n_cols * 8  # float64 = 8 bytes

    if estimated_bytes > MAX_ARRAY_BYTES:
        bam.close()
        print(
            f"[SKIP] {bam_path}: "
            f"array would require {estimated_bytes / 1024**3:.2f} GB "
            f"({n_rows} x {n_cols}), skipping due to memory limit.",
            file=sys.stderr
        )
        return None, None, None

    try:
        methylation_array = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    except (MemoryError, np.core._exceptions._ArrayMemoryError):
        bam.close()
        print(
            f"[SKIP] {bam_path}: NumPy failed to allocate array "
            f"({n_rows} x {n_cols}), skipping.",
            file=sys.stderr
        )
        return None, None, None

    for i, pos in enumerate(genomic_positions):
        for j, read_id in enumerate(read_ids):
            if read_id in methylation_data[pos]:
                methylation_array[i, j] = methylation_data[pos][read_id]

    return methylation_array, genomic_positions, read_ids

#4. Create Methylation Images
def create_methylation_image(methylation_array1, methylation_array2, genomic_positions1, genomic_positions2, bam_filename, output_dir):
    cmap=ListedColormap(['blue', 'grey', 'red']) #blue = -1, red = 1, grey =0 IGV mapping
    region = os.path.splitext(os.path.basename(bam_filename))[0]
    output_filename = os.path.join(output_dir, f"{region}_comparison.png")

    aligned_arrays1, aligned_arrays2, _ = align_genomic_pos(methylation_array1, methylation_array2, genomic_positions1, genomic_positions2)

    aligned_arrays1 = pad_array(aligned_arrays1, (10000, 50))
    aligned_arrays2 = pad_array(aligned_arrays2, (10000, 50))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.005)
    axs[0].imshow(aligned_arrays1.T, cmap=cmap, aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
    axs[0].axis('off')
    axs[1].imshow(aligned_arrays2.T, cmap=cmap, aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
    axs[1].axis('off')

    #save fig to temporary file
    temp_filename = os.path.join(output_dir, f"{region}_temp.png")
    plt.savefig(temp_filename, bbox_inches='tight')
    plt.close(fig)

    # Crop excess white space from the image
    cropped_image = crop_whitespace_from_image(temp_filename)
    cropped_image.save(output_filename)

    os.remove(temp_filename)

    return output_filename

#H:
def pad_array(array, target_shape):
    padded_array = np.full(target_shape, np.nan)
    rows, cols = min(array.shape[0], target_shape[0]), min(array.shape[1], target_shape[1])
    padded_array[:rows, :cols] = array[:rows, :cols]
    return padded_array

#H:
def crop_whitespace_from_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    original_dpi = img.info.get("dpi")

    is_white = np.all(img_array > [245, 245, 245], axis=-1)

    non_white_rows = np.where(~is_white.all(axis=1))[0]
    non_white_cols = np.where(~is_white.all(axis=0))[0]

    if non_white_rows.size > 0 and non_white_cols.size > 0:
        top, bottom = non_white_rows[0]-5, non_white_rows[-1] + 10
        top, bottom = max(0, top), min(img.height, bottom)

        left, right = non_white_cols[0] - 10, non_white_cols[-1] + 10
        left, right = max(0, left), min(img.width, right)
        img_cropped = img.crop((left, top, right, bottom))

        cropped_array = np.array(img_cropped)
        is_white_cropped = np.all(cropped_array > [245, 245, 245], axis=-1)
        non_white_rows_cropped = np.where(~is_white_cropped.all(axis=1))[0]

        gaps = np.diff(non_white_rows_cropped)
        large_gaps = np.where(gaps > 40)[0]

        if len(large_gaps) > 0:
            gap_start = non_white_rows_cropped[large_gaps[0]]
            gap_end = non_white_rows_cropped[large_gaps[0] + 1]

            keep_gap = 40
            trim_top = gap_start + keep_gap // 2
            trim_bottom = gap_end - keep_gap // 2

            trim_top = max(0, min(trim_top, cropped_array.shape[0]))
            trim_bottom = max(0, min(trim_bottom, cropped_array.shape[0]))
            if trim_bottom > trim_top:
                upper_part = cropped_array[:trim_top, :, :]
                lower_part = cropped_array[trim_bottom:, :, :]
                combined = np.vstack([upper_part, lower_part])
                img_cropped = Image.fromarray(combined.astype(np.uint8))
        img_cropped.info["dpi"] = original_dpi
        return img_cropped
    img.info["dpi"] = original_dpi

    return img

#H:
def align_genomic_pos(a1, a2, gp1, gp2):
    all_gps = sorted(set(gp1) | set(gp2))

    pos1_to_idx = {pos: idx for idx, pos in enumerate(gp1)}
    pos2_to_idx = {pos: idx for idx, pos in enumerate(gp2)}

    aligned1 = np.full((len(all_gps), a1.shape[1]), np.nan)
    aligned2 = np.full((len(all_gps), a2.shape[1]), np.nan)

    for idx, pos in enumerate(all_gps):
        if pos in pos1_to_idx:
            aligned1[idx, :] = a1[pos1_to_idx[pos], :]
        if pos in pos2_to_idx:
            aligned2[idx, :] = a2[pos2_to_idx[pos], :]

    return aligned1, aligned2, all_gps

#5. Predict with pre-trained model
def predict_association(model, image_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (images, _) in enumerate(image_loader):
            filename, _ = image_loader.dataset.samples[i]
            images = images.to(device)
            output = model(images)
            prob = torch.softmax(output, dim=1)
            prob_pos = float(prob[0][1])
            prob_neg = float(prob[0][0])
            prediction = 'negative'

            if prob_pos > 0.5:
                prediction = 'positive'

            predictions.append([filename, prediction])
    return predictions

#6. Methylation diffcs stats
def compute_methylation_difference(bam_with, bam_without, mincov=3, mincpgs=3, cpgdist = 50, minlen = 50, mergegap = 500, diffthresh=0.2):
    bam_w, gp1, _ = create_methylation_array(bam_with)
    bam_wo, gp2, _ = create_methylation_array(bam_without)

    a1, a2, all_gps = align_genomic_pos(bam_w, bam_wo, gp1, gp2)
    deltas = []
    site = []
    for i, pos in enumerate(all_gps):
        calls_with = a1[i, ~np.isnan(a1[i, :])]
        calls_wo = a2[i, ~np.isnan(a2[i, :])]

        if len(calls_with) <mincov or len(calls_wo) <mincov:
            continue

        m_with = int(np.sum(calls_with == 1))
        u_with = int(np.sum(calls_with == -1))
        m_wo = int(np.sum(calls_wo == 1))
        u_wo = int(np.sum(calls_wo == -1))

        reads_with = m_with + u_with
        reads_wo = m_wo + u_wo
        if reads_with == 0 or reads_wo == 0:
            continue

        delta = (m_with/reads_with) - (m_wo/reads_wo)

        deltas.append(delta)

        site.append({'pos':pos+1, 'm_with': m_with, 'u_with': u_with, 'm_wo': m_wo, 'u_wo': u_wo, 'delta': delta})
    if not site:
        return []

    deltas = np.asarray(deltas).reshape(-1, 1)
    if len(deltas) < 2:
        for r in site:
            r['is_diff'] = int(abs(r['delta']) > diffthresh)
            r['is_sim'] = int(abs(r['delta']) <= diffthresh)
    else:
        gmms = GaussianMixture(n_components=2, reg_covar=1e-4, max_iter=500, random_state=1).fit(deltas)
        comp_means = gmms.means_.flatten()
        similar_comp = np.argmin(np.abs(comp_means))
        diff_comp = 1 - similar_comp

        labels = gmms.predict(deltas)
        for r, label in zip(site, labels):
            r['is_diff'] = int(label == diff_comp) #true/1 means there's a diff
            r['is_sim'] = int(label == similar_comp)

    diff_regions = []
    sim_regions = []
    cur_dregion, cur_sregion = [site[0]], [site[0]]
    for prev, curr in zip(site, site[1:]):
        if curr['pos'] - prev['pos'] <= cpgdist and curr['is_diff']==1 and prev['is_diff'] == 1:
            cur_dregion.append(curr)
        else:
            if len(cur_dregion) >= mincpgs:
                diff_regions.append(cur_dregion)
            cur_dregion = [curr]

        if curr['pos'] - prev['pos'] <= 500 and curr['is_sim']==1 and prev['is_sim']==1:
            cur_sregion.append(curr)
        else:
            if len(cur_sregion) >= 10:
                sim_regions.append(cur_sregion)
            cur_sregion = [curr]


    if len(cur_dregion) >= mincpgs:
        diff_regions.append(cur_dregion)
    if len(cur_sregion) >= 10:
        sim_regions.append(cur_sregion)

    sim_regions  = [r for r in sim_regions  if (r[-1]['pos'] - r[0]['pos']) >= 500]

    if not sim_regions: #checks that difference is localised to specific regions, not spread across the whole reads (not global)
        return []

    merged_regions = []
    if diff_regions:
        diff_regions = sorted(diff_regions, key=lambda x: x[0]['pos'])
        cur = diff_regions[0]
        for nxt in diff_regions[1:]:
            gap = nxt[0]['pos'] - cur[-1]['pos']
            block_merge = False
            for s in sim_regions:
                if s[0]['pos'] <= nxt[0]['pos'] - 1 and s[-1]['pos'] >= cur[-1]['pos'] + 1: #overlapping similar regions
                    block_merge = True
                    break
            if gap <= mergegap and not block_merge: #merging close regions
                cur.extend(nxt)
            else:
                merged_regions.append(cur)
                cur = nxt
        merged_regions.append(cur)
    else:
        merged_regions = []

    merged_regions = [r for r in merged_regions if (r[-1]['pos'] - r[0]['pos']) >= minlen]

    results = []
    for region in merged_regions:
        start, end = region[0]['pos'], region[-1]['pos']
        m_with = sum(r['m_with'] for r in region)
        u_with = sum(r['u_with'] for r in region)
        m_wo = sum(r['m_wo'] for r in region)
        u_wo = sum(r['u_wo'] for r in region)

        table = np.array([[m_with, u_with], [m_wo, u_wo]])
        _, pval = fisher_exact(table, alternative='two-sided')

        results.append({'start': start, 'end': end, 'delta_mean': np.mean([r['delta'] for r in region]), 'pval': pval})
    if not results:
        return []
    pvals = np.array([x['pval'] for x in results])
    _, pval_corr, _, _ = multitest.multipletests(pvals, method='fdr_bh')

    for i, rs in enumerate(results):
        rs['pval_adj'] = float(pval_corr[i])

    return results

#GTF anno
def build_gtf_interval(gtf_file):
    intervals = dd(Intersecter)

    gtf = pysam.TabixFile(gtf_file)
    for line in gtf.fetch():
        if line.startswith('#'):
            continue
        chrom, source, feature, start, end, score, strand, frame, attributes = line.split('\t')

        if "chr" not in chrom:
            chrom = "chr" + chrom
        start, end = int(start), int(end)
        attr_dict = {}
        for attr in attributes.strip().split(';'):
            if attr:
                key, val = attr.strip().split()[:2]
                key = key.strip()
                val = val.strip().strip('"')
                attr_dict[key] = val
        possible_ids = [f'{feature}_id', 'ccds_id']
        feature_id = 'NA'
        for fid in possible_ids:
            if fid in attr_dict:
                feature_id = attr_dict[fid]
                break

        if 'gene_id' not in attr_dict:
            continue
        if 'gene_name' not in attr_dict:
            attr_dict['gene_name'] = attr_dict['gene_id']
        gene_name = attr_dict['gene_name']

        interval = Interval(start, end, value=(feature, gene_name, feature_id))
        intervals[chrom].add_interval(interval)
    return intervals

def annotate_gtf(chrom, region_string, intervals):
    feature_priority=['exon', 'CDS', 'transcript', 'gene']

    annotations = []
    if not region_string or region_string == 'NA':
        return 'NA'

    regions = region_string.split(';')

    for region in regions:
        if '=' in region:
            coord, _ = region.split('=')
        else:
            continue

        start, end = map(int, coord.split('-'))

        annotation = []
        if chrom in intervals:
            for rec in intervals[chrom].find(start, end):
                annotation.append(rec.value)
        if not annotation:
            annotations.append('NA')
            continue
        if len(annotation) == 1:
            annotations.append(','.join(annotation[0]))
            continue

        #multiple feature hits
        fp = None
        for feature in feature_priority:
            for a in annotation:
                if a[0] == feature:
                    fp = a
                    break
            if fp:
                break
        if not fp:
            annotations.append(','.join(annotation[0]))
        else:
            annotations.append(','.join(fp))

    return ';'.join(annotations)


# CRE anno
def build_CRE_interval(cre_file):
    intervals = dd(Intersecter)

    with open(cre_file) as f:
        for line in f:
            chrom, start, end, name = line.strip().split()[:4]

            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom

            start, end = int(start), int(end)

            intervals[chrom].add_interval(Interval(start, end, value=name))

    return intervals

def annotate_CRE(chrom, region_string, cre_intervals):
    if region_string == 'NA':
        return 'NA'

    regions = region_string.split(";")
    annotations = []

    for region in regions:
        if not region or '=' not in region:
            annotations.append('NA')
            continue

        coord, _ = region.split('=')

        start, end = map(int, coord.split('-'))

        hits = []
        if chrom in cre_intervals:
            for rec in cre_intervals[chrom].find(start, end):
                hits.append(rec.value)

        if hits:
            annotations.append(','.join(hits))
        else:
            annotations.append('NA')

    return ";".join(annotations)


#7. Generate output file
def generate_output(args, predictions, dir, output_file, fdrpval=0.05, diffthresh=0.2):
    output_list = []
    gtf_annotations = []

    for prediction in predictions:
        filename, pred_label = prediction

        parts = os.path.basename(filename).split('_')
        chrom, start, end, ref, alts = parts[0], parts[1], parts[2], parts[3], parts[4]

        fn = os.path.basename(filename).replace("_comparison.png", ".bam")
        bam_with = os.path.join(dir, fn)
        bam_without = bam_with.replace("_with_variant_sorted.bam", "_without_variant_sorted.bam")
        if not os.path.exists(bam_without):
            logger.warning(f"Missing paired bam file: {bam_without}")
            continue
        reads_alt = sum(1 for _ in pysam.AlignmentFile(bam_with, 'rb'))
        reads_ref = sum(1 for _ in pysam.AlignmentFile(bam_without, 'rb'))

        change_meth = compute_methylation_difference(bam_with, bam_without, mincov=args.mincov, mincpgs=args.mincpgs, cpgdist=args.cpgdist, minlen=args.minlen, mergegap=args.mergegap, diffthresh=args.diffthresh)

        if not change_meth:
            affected_regions = "NA"
            delta = "NA"
            pval_min = "NA"
            class_stats = False
        else:
            ranked = sorted(change_meth, key=lambda r: (r['pval_adj'], r['pval']))
            diffm = [r for r in ranked if abs(r['delta_mean']) > diffthresh]

            top5 = diffm[:5]
            if not top5:
                top5 = ranked[:5]
            pvals = [r['pval_adj'] for r in top5]
            affected_regions = ";".join(f"{r['start']}-{r['end']}={r['pval_adj']:.3e}" for r in top5)
            delta = ";".join(f"{r['delta_mean']:.3f}" for r in top5)
            best = min(top5, key=lambda r: r['pval_adj'])
            pval_min = best['pval_adj']
            delta_p = best['delta_mean']
            class_stats = round(pval_min, 2) < fdrpval and abs(delta_p) > diffthresh

        association = 'Negative'
        if class_stats == True and pred_label == 'positive':
            association = 'Positive'
        if class_stats == True and pred_label == 'negative':
            association = 'Positive'
        if class_stats == False and pred_label == 'positive':
             association = 'Amb'

        output_list.append([chrom, start, end, ref, alts, reads_ref, reads_alt, association, affected_regions, delta, pval_min])

    df = pd.DataFrame(output_list, columns=['Chr', 'Start', 'End', 'Ref', 'Alt', 'Reads_with_Ref', 'Reads_with_Alt', 'Prediction', 'Top 5 DMHs',  'Meth-prop-diffcs', 'min-pval'])

    if args.gtf:
        gtf_intervals = build_gtf_interval(args.gtf)
        for row in output_list:
            chrom = row[0] #, start, end, ref, alts, reads_ref, reads_alt, pred, regions, delta, pval = row
            regions = row[8]
            gtf_annot = annotate_gtf(chrom, regions, gtf_intervals)
            gtf_annotations.append(gtf_annot)
        df["GTF annotation"] = gtf_annotations

    if args.cre:
        for cre_file in args.cre:
            cre_annotations = []
            cre_intervals = build_CRE_interval(cre_file)
            for row in output_list:
                chrom = row[0]
                regions = row[8]
                cre_annot = annotate_CRE(chrom, regions, cre_intervals)
                cre_annotations.append(cre_annot)
            base = os.path.basename(cre_file)
            name = os.path.splitext(base)[0]
            col_name = f"{name}_annotation"
            df[col_name] = cre_annotations

    pred_order = ['Positive', 'Negative', 'Amb']
    df['Prediction'] = pd.Categorical(df['Prediction'], categories=pred_order, ordered=True)
    df = df.sort_values(by=['Prediction', 'min-pval'], ascending=[True, True])
    df = df.drop(columns=['min-pval'])

    df.to_csv(output_file, sep='\t', index=False)

# Main Function
def run(args):
    # Create temporary directory
    tmp_dir = os.path.join(args.tmp, str(uuid4()))
    os.makedirs(tmp_dir, exist_ok=True)
    img_dir = os.path.join(tmp_dir, "unknown")
    os.makedirs(img_dir)

    model_path = args.model
    model = Net(224, num_layers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("torch device: %s" % device)

    # Separate reads by variant
    if args.sv:
        separate_reads_by_sv(args.bam_file, args.vcf_file, tmp_dir, args)
    else:
        separate_reads_by_variant(args.bam_file, args.vcf_file, tmp_dir, args)

    # Index BAM files
    index_bam_files(tmp_dir)

    # Process and predict associations
    image_paths = []
    logger.info("Creating methylation arrays and images")
    for bam_file in os.listdir(tmp_dir):
        if bam_file.endswith("_with_variant_sorted.bam"):
            without_variant_file = bam_file.replace("_with_variant_sorted.bam", "_without_variant_sorted.bam")
            if os.path.exists(os.path.join(tmp_dir, without_variant_file)):
                # Generate methylation arrays
                meth_array1, pos1, _ = create_methylation_array(os.path.join(tmp_dir, bam_file), meth_cutoff=args.meth_cutoff, unmeth_cutoff=args.unmeth_cutoff, mem=args.memory)
                meth_array2, pos2, _ = create_methylation_array(os.path.join(tmp_dir, without_variant_file), meth_cutoff=args.meth_cutoff, unmeth_cutoff=args.unmeth_cutoff, mem=args.memory)

                if meth_array1 is not None and meth_array2 is not None:
                    # Create image and add to list
                    img_path = create_methylation_image(meth_array1, meth_array2, pos1, pos2, bam_file, img_dir)
                    if img_path:
                        image_paths.append(img_path)

    # Load images and make predictions
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image_dataset = ImageFolder(tmp_dir, transform=transform)
    image_loader = DataLoader(image_dataset, batch_size=1, shuffle=False)
    logger.info(f"Making predictions...\n")
    predictions = predict_association(model, image_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Parse SNP data and generate output
    output_file = f'{os.path.splitext(os.path.basename(args.vcf_file))[0]}_output.tsv'
    generate_output(args, predictions, tmp_dir, output_file, fdrpval=args.fdrpval, diffthresh=args.diffthresh)

    # Clean up temporary directory
    subprocess.run(["rm", "-rf", tmp_dir])
    logger.info(f"Results saved to {output_file}")

def main():
    args = parse_args()
    run(args)

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Predict variant associations with methylation using a pre-trained model.")

    parser.add_argument("-b", "--bam_file", help="Input BAM file", required=True)
    parser.add_argument("-v", "--vcf_file", help="Input VCF file", required=True)
    parser.add_argument("-c", "--chromosome", help="Limit analysis to a specific chromosome; must be chr3 instead of 3", required=False)
    parser.add_argument("-r", "--region", help="Limit analysis to a specific region in chr:start-end format; chr1:100-200", required=False)
    parser.add_argument("--meth_cutoff", type=float, default=0.8, help="threshold above which a base is considered methylated (default=0.8)")
    parser.add_argument("--unmeth_cutoff", type=float, default=0.2, help="threshold below which a base is considered unmethylated (default=0.2)")
    parser.add_argument("--memory", type=int, default=8, help="maximum GB for methylation array. Avoids very large files from INV and DUP SVs")
    parser.add_argument("--mincpgs", type=int, default=3, help="minimum CpGs to consider before defining a region of potential difference (default=3). Can be up to 10")
    parser.add_argument("--cpgdist", type=int, default=50, help="maximum distance between consecutive CpGs in a differential region (default=50)")
    parser.add_argument("--mergegap", type=int, default=500, help="maximum distance between two identified differential region to merge as one (default=100)")
    parser.add_argument("--mincov", type=int, default=3, help="minimum number of reads for a genomic position to be considered for methylation estimation (default=3)")
    parser.add_argument("--minlen", type=int, default=50, help="minimum length of differential region")
    parser.add_argument("--diffthresh", type=float, default=0.2, help="threshold above which methylation proportion difference is considered big enough (default=0.2)")
    parser.add_argument("--fdrpval", type=float, default=0.05, help="significance cut-off for methylation difference calculated with fisher's test default(0.05)")
    parser.add_argument("--model", help="path to pre-trained model", default=MODEL_PATH)
    parser.add_argument("--tmp", help="temp folder that supports writing of thousands of files, depending on variant size", default=os.getcwd())
    parser.add_argument("--gtf", help="tabix-indexed gtf file")
    parser.add_argument("--cre", action="append", help="file with CREs: promoters, enhancers, repeats etc. No header row; 4 cols of chr, start, end, feature_name")
    parser.add_argument("--sv", action="store_true", help="activate function for structural variants")

    args=parser.parse_args()
    return args

if __name__ == "__main__":
    main()

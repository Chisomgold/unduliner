#!/usr/bin/env python

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import warnings
import pysam
import argparse
import subprocess
from uuid import uuid4
from PIL import Image

import logging
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def separate_reads_by_variant(bam_file, vcf_file, output_dir):
    logger.info("reading input files")
    bam = pysam.AlignmentFile(bam_file, 'rb')
    vcf_reader = pysam.VariantFile(vcf_file)

    for record in vcf_reader.fetch():
        chrom = record.chrom
        start = record.pos - 1
        end = start + len(record.ref)
        alt = record.alts[0]
        ref = record.ref

        # Create output files
        with_variant = pysam.AlignmentFile(f'{output_dir}/{chrom}_{start}_{end}_with_variant.bam', 'wb', template=bam)
        without_variant = pysam.AlignmentFile(f'{output_dir}/{chrom}_{start}_{end}_without_variant.bam', 'wb', template=bam)

        logger.info(f'parse {chrom}:{start}-{end} {ref}/{alt}')

        reads = bam.fetch(chrom, start, end)
        for read in reads:
            if read.query_sequence is None:
                continue
            for positions in read.get_aligned_pairs():
                if positions[0] is None:
                    continue
                if positions[1] == start:
                    if read.query_sequence[positions[0]] == alt:
                        with_variant.write(read)
                        logger.debug(f"wrote {read.query_name} to with_variant; alt, ref is {alt}, {ref}; read has {read.query_sequence[positions[0]]}")
                    elif read.query_sequence[positions[0]] == ref:
                        without_variant.write(read)
                        logger.debug(f"wrote {read.query_name} to without_variant; alt, ref is {alt}, {ref}; read has {read.query_sequence[positions[0]]}")

        with_variant.close()
        without_variant.close()

    bam.close()
    vcf_reader.close()

def index_bam_files(output_dir):
    logger.info("Indexing bam files")
    for bam_file in os.listdir(output_dir):
        if bam_file.endswith(".bam"):
            bam_file_path = os.path.join(output_dir, bam_file)
            sorted_bam = bam_file_path.replace(".bam", "_sorted.bam")

            if not os.path.exists(sorted_bam):
                subprocess.run(["samtools", "sort", "-o", sorted_bam, bam_file_path])
            subprocess.run(["samtools", "index", sorted_bam])

#Extract methylation data
def create_methylation_array(bam_path):
    bam = pysam.AlignmentFile(bam_path, "rb")
    methylation_data = {}

    for read in bam:
        if read.query_sequence is None:
            continue
        read_id = read.query_name
        aligned_positions = {query_pos: ref_pos for query_pos, ref_pos in read.get_aligned_pairs(matches_only=True)}

        for key, values in read.modified_bases.items():
            for base_pos, quality in values:
                quality_div = quality/256
                if quality_div > 0.8:
                    probability=1
                elif quality_div < 0.2:
                    probability = -1
                else:
                    probability = 0
                if base_pos in aligned_positions:
                    ref_pos = aligned_positions[base_pos]
                    if ref_pos not in methylation_data:
                        methylation_data[ref_pos] = {}
                    methylation_data[ref_pos][read_id] = probability

    genomic_positions = sorted(methylation_data.keys())
    read_ids = list({read_id for pos_data in methylation_data.values() for read_id in pos_data})
    np.set_printoptions(threshold=np.inf)
    methylation_array = np.full((len(genomic_positions), len(read_ids)), np.nan)

    for i, pos in enumerate(genomic_positions):
        for j, read_id in enumerate(read_ids):
            if read_id in methylation_data[pos]:
                methylation_array[i, j] = methylation_data[pos][read_id]

    bam.close()
    return methylation_array, genomic_positions, read_ids

#Create methylation image
def create_methylation_image(methylation_array1, methylation_array2, genomic_positions1, genomic_positions2, bam_filename, output_dir):
    cmap=ListedColormap(['blue', 'grey', 'red']) #blue = -1, red = 1, grey =0 IGV mapping
    region = os.path.splitext(os.path.basename(bam_filename))[0]
    output_filename = os.path.join(output_dir, f"{region}_flipped.png")

    if len(methylation_array1.T) < 3 or len(methylation_array2.T) < 3: #3 reads minimum
        return None

    if len(methylation_array1.T) < 0.3 * len(methylation_array2.T) or len(methylation_array2.T) < 0.3 * len(methylation_array1.T):
        print(f'Sequence depth less than 30% of corresponding pair. Skipping {bam_filename}')
        return None

    aligned_arrays1, aligned_arrays2 = align_genomic_pos(methylation_array1, methylation_array2, genomic_positions1, genomic_positions2)

    aligned_arrays1 = pad_array(aligned_arrays1, (10000, 50))
    aligned_arrays2 = pad_array(aligned_arrays2, (10000, 50))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.005)
    axs[0].imshow(aligned_arrays2.T, cmap=cmap, aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
    axs[0].axis('off')
    axs[1].imshow(aligned_arrays1.T, cmap=cmap, aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
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

#H: Pad Array
def pad_array(array, target_shape):
    padded_array = np.full(target_shape, np.nan)
    rows, cols = min(array.shape[0], target_shape[0]), min(array.shape[1], target_shape[1])
    padded_array[:rows, :cols] = array[:rows, :cols]
    return padded_array

#H: crop image to reduce Whitespace; remove rows all white
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

    return aligned1, aligned2

def main(args):
    dir=os.path.join("/lustre/user/", "IgvCmapimages")
    os.makedirs(dir, exist_ok=True)
    img_dir=os.path.join(dir, "neg")
    os.makedirs(img_dir)

    # Separate reads by variant
    separate_reads_by_variant(args.bam_file, args.vcf_file, dir)

    # Index BAM files
    index_bam_files(dir)

    subprocess.run(["find", dir, "-name", "*variant.bam", "-exec", "rm", "-f", "{}", "+"])

    logger.info("Creating methylation arrays and images")
    for bam_file in os.listdir(dir):
        if bam_file.endswith("_with_variant_sorted.bam"):
            without_variant_file = bam_file.replace("_with_variant_sorted.bam", "_without_variant_sorted.bam")
            if os.path.exists(os.path.join(dir, without_variant_file)):
                # Generate methylation arrays
                meth_array1, pos1, _ = create_methylation_array(os.path.join(dir, bam_file))
                meth_array2, pos2, _ = create_methylation_array(os.path.join(dir, without_variant_file))

                if meth_array1 is not None and meth_array2 is not None:
                    create_methylation_image(meth_array1, meth_array2, pos1, pos2, bam_file, img_dir)

    subprocess.run(["find", dir, "-name", "*.bam", "-exec", "rm", "-f", "{}", "+"])
    subprocess.run(["find", dir, "-name", "*.bai", "-exec", "rm", "-f", "{}", "+"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image gen')
    parser.add_argument("-b", "--bam_file", help="Input BAM file", required=True)
    parser.add_argument("-v", "--vcf_file", help="Input VCF file", required=True)

    args=parser.parse_args()
    main(args)

#!/usr/bin/bash

awk '{l=$9; gsub(/=[^;]+/,"",l); gsub(/;/,",",l); print $1":"$2+1, $1":"($2-30000)"-"($3+30000), l}' regionsforplotting.txt > regs
zgrep -v ^# testdata/chr17atcc.vcf.gz | awk '{print $1":"$2,$3}' > vcf.ids

awk 'NR==FNR{ids[$1]=$2; next} {split($1,a,"-"); key=a[1]; if (key in ids) print $2, $3, ids[key]}' vcf.ids regs | while read -r interval highlight id

do
methylartist locus -b /lustre/thatanda/MCF7/stats/ATCC.haplotag.bam -i $interval -l $highlight -m m --motif CG --ref /lustre/taewing/ref/hg38/Homo_sapiens_assembly38.fasta --variants testdata/chr17atcc.vcf.gz --splitvar $id --gtf /lustre/thatanda/MCF7/file.sorted.gtf.gz --labelgenes --skip_raw_plot --highlight_alpha 0.7 --width 30

done

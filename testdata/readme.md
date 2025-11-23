# Test data

This folder contains example datasets for testing and demonstrating the functionality of the tool using the example code provided. 


The images were created using `methylartist locus` command from the methylartist package with the variant separation feature. IGV screenshots are added to show regions
highlighted as differentially methylated in the positive predicted variant; alignments are grouped by base at variant position.

The output from this dataset predicts 2 variants to be positive and others as negative or ambiguous. There is an additional file called 
`read_parse_summary.txt` with information about which variants have 0 reads and were therefore omitted for prediction.


### Methylartist view of 2 predicted positive variants
- Variant 21303244_C_T
![Methylartist view of predicted positive variant 1:chr17_21303244_C_T_rs736103](chr17_21303244_C_T_rs736103.png)

- Variant 22285282_A_G
![Methylartist view of predicted positive variant 2:chr17_22285282_A_G_rs8065399](chr17_22285282_A_G_rs8065399.png)


### IGV view of regions from predicted positive variants
- Variant 21303244_C_T; Ref reads show higher methylation
![IGV view of positive variant 1:](chr17_21303244_C_T.png)

 - Variant 22285282_A_G; Ref reads show lower methylation
![IGV view of positive variant 2:](chr17_22285282_A_G.png)

### Methylartist view of a predicted negative variant

![Methylartist view of predicted negative variant 1:chr17_20000172_A_C_rs75128113](chr17_20000172_A_C_rs75128113.png)


### Methylartist view of 2 predicted ambiguous variants
- Variant 20002512_A_G
![Methylartist view of ambiguous variant 1:chr17_20002512_A_G_rs35315252.png](chr17_20002512_A_G_rs35315252.png)

- Variant 20002618_C_T

![Methylartist view of predicted ambiguous variant 2:chr17_20002618_C_T_rs79773274](chr17_20002618_C_T_rs79773274.png)





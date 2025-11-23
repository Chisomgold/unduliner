# Test data

This folder contains example datasets for testing and demonstrating the functionality of the main tool. 


The images are created using `methylartist locus` command from the methylartist package with the variant separation feature. IGV screenshots are added to show regions
highlighted as differentially methylated in the positive predicted variant; alignments are grouped by base at variant position.

The output from this dataset predicts 2 variants to be positive and others as negative or ambiguous. There is an additional file called 
`read_parse_summary.txt` with information about which variants has 0 reads and were therefore omitted for prediction.


### Methylartist view of 2 predicted positive variants
![Methylartist view of predicted positive variant 1:chr17_21303244_C_T_rs736103](chr17_21303244_C_T_rs736103.png)

![Methylartist view of predicted positive variant 2:chr17_22285282_A_G_rs8065399](chr17_22285282_A_G_rs8065399.png)


### IGV view of regions from predicted positive variants
![IGV view of positive variant 1:](chr17_21303244_C_T.png)

![IGV view of positive variant 2:](chr17_22285282_A_G.png)

### Methylartist view of 2 predicted negative variants

![Methylartist view of predicted negative variant 1:chr17_20000172_A_C_rs75128113](chr17_20000172_A_C_rs75128113.png)

![Methylartist view of predicted negative variant 2:chr17_20002618_C_T_rs79773274](chr17_20002618_C_T_rs79773274.png)


### Methylartist view of 2 predicted ambiguous variants

![Methylartist view of ambiguous negative variant 1:chr17_20002512_A_G_rs35315252.png](chr17_20002512_A_G_rs35315252.png.png)

![Methylartist view of ambiguous negative variant 2:chr17_20004614_A_C_rs12944084](chr17_20004614_A_C_rs12944084.png)




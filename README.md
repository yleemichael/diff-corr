# diff-corr

####Differential correlation analysis (pairwise) to identify significant connections in a complex network.

*Written in Python using NumPy and Pandas.
Optimized with vectorization.*

Two groups: Control and Case.

Control is defined as a 'Group' value of 0. Case is 'Group' value of 1.
The Group column should always be the first column in the input file.

The input data file should be in .csv and the following format:

    Group,Glycine,Arginine,Taurine
    0,21,0.103,1002
    0,32,0.215,3005
    0,16,0.089,2516
    0,25,0.158,4122
    0,27,0.201,3532
    1,46,0.018,2118
    1,78,0.009,3157
    1,52,0.017,4129
    1,61,0.009,1059
    1,58,0.015,3353
    1,51,0.016,2798

Columns are features/variables of interest. Rows are samples/individuals.

The output file will be in .csv and the following format:

    Metabolite_1,Metabolite_2,Control_Corr,Control_p,Case_Corr,Case_p,Diff_Corr,Diff_Corr_p,Sign
    Glycine,Arginine,-0.8635874080236875,0.026643383160278512,0.9544892812818188,0.011574899681934028,3.4810615582681352,0.0,-1
    Glycine,Taurine,0.03452369355413695,0.9482350338122344,0.44473429104479945,0.45301028954404177,0.43581744661388694,628.0,-1
    Arginine,Taurine,0.40908713722203893,0.42060012792797347,0.6059124510103757,0.27875138351537426,0.17026786955232198,825.0,-1

The following output variables are given for each pair of features (columns) in case and control respectively:
- Control_Corr - Pearson correlation coefficient r of two features/columns in the control group
- Control_p - p value of the Pearson correlation coefficient for control group
- Case_Corr - Pearson correlation coefficient r of two features/columns in the case group
- Case_p - p value of the Pearson correlation coefficient for case group
- Diff_Corr - differential correlation coefficient between case and control for same pair of features/columns (sign may be removed with absolute value function depending on bool argument provided)
- Diff_Corr_p - p value of Diff_Corr, multiplied by 1000 for ease of use in network visualization programs. Calculated using random permutation analysis.
- Sign - positive or negative sign associated with Diff_Corr (can be turned off in arguments)

Code Execution

`differential_correlation_network_analysis(filename, total_permutations)`

`differential_correlation_network_analysis('sample_data', 1000)`

Three arguments are required.
- filename - located in `data/` directory (e.g. `'sample_data'`).
- total_permutations - 1000 is standard to attain a precise p value. Can increase or decrease at your discretion.
Increasing will lengthen runtime. Decreasing will reduce precision.

This program was written for a metabolomics research project on Alzheimer's Disease. As an example, the columns (features) in this project were different metabolites concentrations
and the rows (samples) were patients. By using this program, I could visualize a network of connections between different metabolites
and identify significant relationships. Thus, this research project could help determine which metabolites to focus on in biological wet-lab
experimentation.

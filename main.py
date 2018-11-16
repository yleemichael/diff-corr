import networkx as nx
import pandas as pd
import numpy as np
from random import randint
from scipy.stats import pearsonr

def separate_groups(df_raw):
    #Initiate Control (df[0]) and Case (df[1]) DataFrames.
    df_list = [pd.DataFrame(columns=df_raw.columns), pd.DataFrame(columns=df_raw.columns)]

    #Add df_raw group 0 rows to df[0] and df_raw group 1 rows to df[1].
    for n, row in df_raw.iterrows():
        lrow = list(row)
        df_list[int(lrow[0])] = pd.concat([df_list[int(lrow[0])],pd.DataFrame([row])],ignore_index=True)

    for i, df in enumerate(df_list):
        # Remove Case/Control Designation (1st column) since it is irrelevant for correlation analysis
        df.drop(df.columns[0], axis=1, inplace=True)

    return df_list


'''
Mean Normalization
	Value := (Value - Mean_of_Column) / Range_of_Column
	Range or St. Dev ^ 2
'''
def mean_normalization(df_list):
    for j in range(1,len(df_list[0].columns)):
        combined_column_list = np.array(list(df_list[0].iloc[:,j]))
        np.append(combined_column_list, np.array(list(df_list[1].iloc[:,j])))
        col_mean = np.mean(combined_column_list)
        #col_range = (df_list[0].iloc[:,j] + df_list[1].iloc[:,j]).max() - (df_list[0].iloc[:,j] + df_list[1].iloc[:,j]).min()
        col_stdev = np.std(combined_column_list)
        for i in range(len(df_list)):
            #df_list[i].iloc[:, j] = (df_list[i].iloc[:, j] - col_mean) / col_range
            df_list[i].iloc[:, j] = (df_list[i].iloc[:, j] - col_mean) / (col_stdev ** 2)

    return df_list


'''
Pearson's correlation coefficient between all possible feature pairs (within each group)

	Create two lists that contain the r values for control and case, respectively.
	Order of lists is significant, as r at index x in the control list will match the feature pair of r at same index x in the case list. The only difference is the group designation. This will allow for convenient and efficient comparisons of correlation values between case and control in the future.
'''
def pearsons_correlations(df_list):
    # Two DataFrames of correlations in the list. Index 0 and 1 are for control and case correlations, respectively.
    dfs_r = []

    # Calculate corr values vectorized.
    for i, df in enumerate(df_list):
        dfs_r.append(df.corr())
        dfs_r[i].reset_index(drop=True, inplace=True)

    return dfs_r

def get_pearsons_p_value(df_list):
    arrs_r_p = [[], []]

    # Calculate corr p values non vectorized.
    for df_list_index in range(len(df_list)):
        # Iterate first columns. Index 1 to End columns.
        for col_1_index, col_1_name in enumerate(df_list[df_list_index]):
            col_1 = df_list[df_list_index][col_1_name]

            # First column is group designation and reaching the last column would mean all pairings are completed.
            if col_1_index == len(df_list[df_list_index].columns):
                continue

            # Iterate second columns based on first column position. Always first column's index + 1 until end column.
            for col_2_name in df_list[df_list_index].iloc[:, col_1_index + 1:]:
                col_2 = df_list[df_list_index][col_2_name]
                arrs_r_p[df_list_index].append(pearsonr(col_1, col_2)[1])

    return arrs_r_p

'''
Fischer's z-transformation of Pearson's correlation coefficient, r, of feature pairs
	Transform all r values to z-scores.
'''
def fischers_ztransformation(dfs_r):
    # Two numpy arrays of z scores in the list. Index 0 and 1 are for control and case z scores, respectively.
    nparrs_z = []

    for i, df_r in enumerate(dfs_r):
        numerator = 1 + np.array(df_r)
        denominator = 1 - np.array(df_r)
        log_argument = np.divide(numerator, denominator, out=numerator/0.0000000001, where=denominator!=0)
        nparrs_z.append(0.5*np.log(log_argument, out=log_argument/0.0000000001, where=log_argument!=0))
    return nparrs_z

'''
Determine the n value of case and control groups.
	This function assumes that all columns within each group contain the same number of values.
'''
def get_n(df_list):
    # (n_control, n_case)
    return (len(df_list[0].index), len(df_list[1].index))


'''
Differential correlations, r_diff, of feature pairs
	Compare the correlations of feature pairs between control and case conditions.
'''
def differential_correlations(nparrs_z, n_tuple):
    n_control = n_tuple[0]
    n_case = n_tuple[1]

    nparr_dc = np.sqrt((n_case - 3) / 2) * nparrs_z[1] - np.sqrt((n_control - 3) / 2) * nparrs_z[0]

    return nparr_dc

'''
P value determination - random permutation analysis
	1000 permutations
'''
def get_p_values(df_list, nparr_dc, n_tuple, total_permutations):
    n_control = n_tuple[0]
    n_case = n_tuple[1]
    total_swaps = int(np.round((n_case * n_control) / (n_case + n_control)))
    nparr_p = np.zeros((np.size(nparr_dc, 0), np.size(nparr_dc, 1)))

    for i, permutation_n in enumerate(range(total_permutations)):
        df_list_copy = df_list.copy()

        for n_swap in range(total_swaps):
            random_control_index = randint(0, n_control - 1)
            random_case_index = randint(0, n_case - 1)

            # Copy random control and case sample whole row into temp_control and temp_case
            temp_control = df_list_copy[0].iloc[random_control_index, :].copy()
            temp_case = df_list_copy[1].iloc[random_case_index, :].copy()

            # Switch the whole row and then switch again the labels of each new row
            df_list_copy[0].iloc[random_control_index, :] = temp_case
            df_list_copy[0].iloc[random_control_index, 0] = temp_control[0]

            df_list_copy[1].iloc[random_case_index, :] = temp_control
            df_list_copy[1].iloc[random_case_index, 0] = temp_case[0]

        # e.g. (r, p)
        dfs_r = pearsons_correlations(mean_normalization(df_list))
        nparr_dc_copy = differential_correlations(fischers_ztransformation(dfs_r), n_tuple)

        nparr_temp_bool = np.absolute(nparr_dc) < np.absolute(nparr_dc_copy)
        nparr_temp_int = nparr_temp_bool.astype(int)
        nparr_p += nparr_temp_int

    nparr_p /= total_permutations

    return nparr_p


'''
Create the Complex Network using NetworkX
'''
def create_network(df_list):
    # df = dataframe, dc = differential correlations
    # Determine edges and edge weights.

    G = nx.Graph()

    for feature_1_index, feature_1_name in enumerate(df_list[0]):
        # Skip last column because no pairs are possible.
        if feature_1_index == len(df_list[0].columns):
            continue
        for feature_2_name in df_list[0].iloc[:,feature_1_index + 1:]:
            G.add_edge(feature_1_name, feature_2_name)

    return G.edges()

def create_df_output(edges, dfs_r, arrs_r_p, nparr_dc, nparr_p, isolate_diff_corr_signs, total_permutations):
    df_output = pd.DataFrame(columns=['Metabolite_1', 'Metabolite_2', 'Control_Corr', 'Control_p', 'Case_Corr', 'Case_p', 'Diff_Corr', 'Diff_Corr_p', 'Sign'])
    for i in range(len(edges)):
        df_output.loc[i] = [0 for n in range(len(df_output.columns))]

    for i, edge in enumerate(edges):
        # Metabolite1 and Metabolite2
        df_output.iloc[i, 0] = edge[0]
        df_output.iloc[i, 1] = edge[1]

    for i, df_r in enumerate(dfs_r):
        added_count = 0
        for j, row in df_r.iterrows():
            skip_count = 1 + j
            for k, r in enumerate(row):
                if skip_count > 0:
                    skip_count -= 1
                    continue
                else:
                    # Case pearsons r and p
                    if i == 1:
                        df_output.iloc[added_count, 2] = r
                    # Control pearsons r and p
                    else:
                        df_output.iloc[added_count, 4] = r
                added_count += 1

    for i, nparr_r_p in enumerate(arrs_r_p):
        for j, p in enumerate(nparr_r_p):
            # Case pearsons p
            if i == 1:
                df_output.iloc[j, 3] = p
            # Control pearsons p
            elif i == 0:
                df_output.iloc[j, 5] = p

    added_count = 0
    for i, row in enumerate(nparr_dc):
        skip_count = 1 + i
        for j, dc in enumerate(row):
            if skip_count > 0:
                skip_count -= 1
                continue
            # Differential Correlation Coefficients
            if not isolate_diff_corr_signs:
                df_output.iloc[added_count, 6] = dc
            else:
                if dc >= 0:
                    df_output.iloc[added_count, 6] = dc
                    df_output.iloc[added_count, 8] = 1
                else:
                    df_output.iloc[added_count, 6] = np.absolute(dc)
                    df_output.iloc[added_count, 8] = -1
            added_count += 1

    added_count = 0
    for i, row in enumerate(nparr_p):
        skip_count = 1 + i
        for j, p in enumerate(row):
            if skip_count > 0:
                skip_count -= 1
                continue
            # Differential Correlation p values
            df_output.iloc[added_count, 7] = p * total_permutations
            added_count += 1

    return df_output

def write_df_output(filename, df_output):
    df_output.to_csv('output/' + filename + '_OUTPUT.csv', index=False)

def write_df_sig_output(filename, df_output, significance_level, total_permutations):
    '''
    The output p values have been multiplied by 1000 for network visualization programs such as Cytoscape.
    E.g. 0.05 would output as 50.
    '''
    df_sig_output = pd.DataFrame(columns=['Metabolite_1', 'Metabolite_2', 'Control_Corr', 'Control_p', 'Case_Corr', 'Case_p', 'Diff_Corr', 'Diff_Corr_p', 'Sign'])
    for index, row in df_output.iterrows():
        if df_output.iloc[index, 7] <= significance_level * total_permutations:
            df_sig_output = df_sig_output.append(row, ignore_index=True)

    df_sig_output.to_csv('output/' + filename + '_OUTPUT_SIG.csv', index=False)

'''
Run Differential Correlation Network Analysis
'''
def differential_correlation_network_analysis(filename, total_permutations, significance_level=0.05, isolate_diff_corr_signs=False, output_isolated_significant=False):
    '''
    `filename` must be in .csv.

    `significance_level` ranges from 0 and 1. However, the output file will display the p value as the true value multiplied by
    1000 for usage in network visualization programs, such as Cytoscape, which are unable to process significant values
    within the 0 to 1 scale and requires a larger whole number scale.

    Setting `isolate_diff_corr_signs` to True will result in a 9th column (index 8) in the final output file
    containing the sign of the differential correlation coefficients (represented as -1 and 1).
    Additionally, the negative signs are removed from the 7th column (index 6) with the absolute value function.
    This allows for edge color differentiation based on sign in various network visualization programs, such as Cytoscape.

    Setting `output_isolated_significant` to True will provide an additional output file that only contains differential
    correlation values that are significant.
    '''

    df_raw = pd.read_csv('data/' + filename + '.csv')
    df_list = separate_groups(df_raw)
    n_tuple = get_n(df_list)

    df_mean_normalized = mean_normalization(df_list)
    dfs_r = pearsons_correlations(df_mean_normalized)
    arrs_r_p = get_pearsons_p_value(df_mean_normalized)

    nparr_dc = differential_correlations(fischers_ztransformation(dfs_r), n_tuple)
    nparr_p = get_p_values(df_list, nparr_dc, n_tuple, total_permutations)
    edges = create_network(df_list)

    df_output = create_df_output(edges, dfs_r, arrs_r_p, nparr_dc, nparr_p, isolate_diff_corr_signs, total_permutations)

    write_df_output(filename, df_output)

    if output_isolated_significant:
        write_df_sig_output(filename, df_output, significance_level, total_permutations)

differential_correlation_network_analysis('sample_data', 1000, 0.05, True, True)

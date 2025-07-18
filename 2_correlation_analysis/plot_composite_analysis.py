import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def plot_from_person_level_data():
    # Read the CSV
    df = pd.read_csv('composite_outcomes_person_level_correlations.csv')

    # Define inputs and categories
    inputs = ['ALLCONDITIONS', 'ENVIRONMENTAL', 'AMERICANVOICES', 'DEMOGRAPHIC']
    label_map = {
        'ALLCONDITIONS': 'Combined',
        'AMERICANVOICES': 'American Voices Project',
        'ENVIRONMENTAL': 'Climate',
        'DEMOGRAPHIC': 'Demographic'
    }
    n_inputs = len(inputs)
    categories = ['env', 'ind']
    category_labels = ['Climate', 'Individual Difference']

    # Set colors for inputs
    colors = {
        'ALLCONDITIONS': 'grey',
        'ENVIRONMENTAL': 'skyblue',
        'AMERICANVOICES': 'salmon',
        'DEMOGRAPHIC': 'lightgreen'
    }

    # Prepare the plot data
    bar_means = []
    bar_errors = []

    for category in categories:
        for input_type in inputs:
            column_name = f'{input_type}_waves_{category}_correlation'
            if column_name in df.columns:
                values = df[column_name].dropna()
                mean = values.mean()
                sem = values.sem()  # Standard Error of the Mean
                bar_means.append(mean)
                bar_errors.append(sem)
            else:
                bar_means.append(np.nan)
                bar_errors.append(np.nan)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Number of categories
    n_categories = len(categories)

    # Positions of groups on x-axis
    index = np.arange(n_categories)

    # Width of each bar
    bar_width = 0.2

    # Offsets for each input within a category
    offsets = {
        'ALLCONDITIONS': -2*bar_width,
        'ENVIRONMENTAL': -bar_width,
        'AMERICANVOICES': 0,
        'DEMOGRAPHIC': bar_width
    }

    # Plot each set of bars
    for i, input_type in enumerate(inputs):
        means = bar_means[i::n_inputs]
        errors = bar_errors[i::n_inputs]
        ax.bar(index + offsets[input_type], means, bar_width,
            label=label_map.get(input_type, input_type),
            color=colors[input_type], yerr=errors, capsize=5)

    # Set labels and title
    ax.set_xlabel('Measure Categories')
    ax.set_ylabel('Average Correlation')
    ax.set_title('Individual-Level Average Correlation by Input Type and Category')
    ax.set_xticks(index)
    ax.set_xticklabels(category_labels)
    ax.legend(title='Agent Type', loc='lower left')

    plt.tight_layout()
    plt.show()


def plot_from_measures_level():

    measures = {
        "adaptation_mitigation_output.csv": {
            "cat": "env",
            "cols": [
                {"pred": "Overall_Pred", "w2": "Truth_Overall_W2" , "true": "Truth_Overall_W1"}
            ]
        },
        "cns_output.csv": {
            "cat": "env",
            "cols": [
                {"pred": "CNS_Composite_Pred", "w2": "CNS_Composite_W2", "true": "CNS_Composite_W1"}
            ]
        },
        "ecdc_output.csv": {
            "cat": "env",
            "cols": [
                {"pred": "Collective_Pred", "w2": "Collective_W2", "true": "Collective_W1"},
                {"pred": "Indiv_Pred", "w2": "Indiv_W2", "true": "Indiv_W1"}
            ]
        },
        "envefficacy_match.csv": {
            "cat": "env",
            "cols": [
                {"pred": "Eff_Indiv_Pred", "w2": "Eff_Indiv_W2", "true": "Eff_Indiv_W1"},
            {"pred": "Eff_Collective_Pred", "w2": "Eff_Collective_W2" , "true": "Eff_Collective_W1"}
            ]
        },
        "envactions_output.csv": {
            "cat": "env",
            "cols": [{"pred": "ENV_ACTIONS_Composite_Pred", "w2":"ENV_ACTIONS_Composite_W2", "true": "ENV_ACTIONS_Composite_W1"}]
        }, 
        "gses_output.csv": {
            "cat": "ind",
            "cols": [
                {"pred": "GSE_Composite_Pred", "w2": "GSE_Composite_W2", "true": "GSE_Composite_W1"}
            ]
        },
        "gsjs_output.csv": {
            "cat": "ind",
            "cols": [
                {"pred": "GSJS_Composite_Pred", "w2":  "GSJS_Composite_W2", "true": "GSJS_Composite_W1"}
            ]
        },
        "iri_output.csv": {
            "cat": "ind",
            "cols": [
                {"pred": "IRI_PT_Pred", "w2": "IRI_PT_W2", "true": "IRI_PT_W1"},
                {"pred": "IRI_FS_Pred", "w2": "IRI_FS_W2", "true": "IRI_FS_W1"},
                {"pred": "IRI_EC_Pred", "w2": "IRI_EC_W2", "true": "IRI_EC_W1"},
                {"pred": "IRI_PD_Pred", "w2": "IRI_PD_W2", "true": "IRI_PD_W1"}
            ]
        },
        "mfq_output.csv": {
            "cat": "ind",
            "cols": [
                {"pred": "MFQ_overall_Pred", "w2": "MFQ_overall_W2", "true": "MFQ_overall_W1"}
            ]
        },
        "nep_output.csv": {
            "cat": "env",
            "cols": [
                {"pred": "NEP_Composite_Pred", "w2": "NEP_Composite_W2" , "true": "NEP_Composite_W1"}
            ]
        },
        "nfc_output.csv": {
            "cat": "ind",
            "cols": [
                {"pred": "NFC_Composite_Pred", "w2": "NFC_Composite_W2", "true": "NFC_Composite_W1"}
            ]
        },
        "proximity_output.csv": {
            "cat": "env",
            "cols": [
                {"pred": "Proximity_Pred", "w2": "Proximity_W2", "true": "Proximity_W1"}
            ]
        },
        "risk_aversion_output.csv": {
            "cat": "ind",
            "cols": [
                {"pred": "RA_Composite_Pred", "w2": "RA_Composite_W2", "true": "RA_Composite_W1"}
            ]
        },
        "trust_output.csv": {
            "cat": "env",
            "cols": [
                {"pred": "Trust_Composite_Pred", "w2": "Trust_Composite_W2", "true": "Trust_Composite_W1"}
            ]
        },
        "sdo_output.csv": {
            "cat": "ind",
            "cols": [
                {"pred": "SDO_Composite_Pred", "w2":  "SDO_Composite_W2", "true": "SDO_Composite_W1"}
            ]
        },
        "mes_composite_output.csv": {
            "cat": "ind",
            "cols": [
                {"pred": "MES_Composite_Pred", "w2":  "MES_Composite_W2", "true": "MES_Composite_W1"}
            ]
        },
            "emotions_output.csv": {
            "cat": "env",
            "cols": [
                {"pred": "Envemotions_Composite_Pred", "w2":  "Envemotions_Composite_W2", "true": "Envemotions_Composite_W1"}
            ]
        },
    }

    # Read the CSV
    df = pd.read_csv('composite_outcomes_correlations.csv', index_col=0)

    # Inputs and colors
    inputs = ['ALLCONDITIONS', 'ENVIRONMENTAL', 'AMERICANVOICES', 'DEMOGRAPHIC']
    label_map = {
        'ALLCONDITIONS': 'Combined',
        'AMERICANVOICES': 'American Voices Project',
        'ENVIRONMENTAL': 'Climate',
        'DEMOGRAPHIC': 'Demographic'
    }
    input_colors = {
        'ALLCONDITIONS': 'grey',
        'ENVIRONMENTAL': 'skyblue',
        'AMERICANVOICES': 'salmon',
        'DEMOGRAPHIC': 'lightgreen'
    }

    # Categories mapping
    #categories = {'env': 'Environmental', 'soc': 'Social', 'ind': 'Individual'}
    categories = {'env': 'Climate', 'ind': 'Individual Difference'}

    # Initialize storage for each (category, input)
    data = {cat: {input_: [] for input_ in inputs} for cat in categories.keys()}

    # Process the data
    for index in df.index:
        for filename, info in measures.items():
            for col_pair in info['cols']:
                pred_col = col_pair['pred']
                true_col = col_pair['true']
                expected_string = f"{filename.split('.csv')[0]}_{pred_col}_vs_{true_col}"
                if expected_string == index:
                    for input_ in inputs:
                        colname = f"{input_}_waves_correlation"
                        value = df.loc[index, colname]
                        if not pd.isna(value):
                            data[info['cat']][input_].append(value)

    # Now calculate means and standard errors
    means = {cat: [] for cat in categories.keys()}
    errors = {cat: [] for cat in categories.keys()}

    for cat in categories.keys():
        for input_ in inputs:
            values = data[cat][input_]
            if values:
                mean = np.mean(values)
                error = np.std(values, ddof=1) / np.sqrt(len(values))  # Standard error
            else:
                mean = np.nan
                error = np.nan
            means[cat].append(mean)
            errors[cat].append(error)

    # Plotting
    x = np.arange(len(categories))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, input_ in enumerate(inputs):
        
        bar = ax.bar(x + idx * width - width,
                [means[cat][idx] for cat in categories.keys()],
                width,
                yerr=[errors[cat][idx] for cat in categories.keys()],
                label=label_map.get(input_, input_),
                color=input_colors[input_],
                capsize=5)


    # Labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Correlation')
    ax.set_xlabel('Measure Categories')
    ax.set_title('Measure-Level Average Correlation by Input Type and Category')
    ax.set_xticks(x)
    ax.set_xticklabels([categories[cat] for cat in categories.keys()])
    ax.legend(title='Agent Type', loc='lower left')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_from_measures_level()
    plot_from_person_level_data()



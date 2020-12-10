# brenda_parse
A repository designated to parsing entries from the BRENDA database

## Installation

Conda requirements to run this parse can be found in environment.yml

## Outputs

This script was run as intended to parse BRENDA and standardized (see sample call below). Both the data and resulting outputs can be found at this [link](https://www.dropbox.com/sh/uh9oqbzauto7o1b/AADYWLlpqFyPtavphAwsT2Nsa?dl=0). 

## Downloads reqired:

This program relies on two different files being downloaded. 

- First, a recent copy of the brenda database can be downloaded at their website [here](https://www.brenda-enzymes.org/download_brenda_without_registration.php). This is the `brenda-flat-file` to include. This can be added to the data folder, `data/raw/brenda_download.txt`. 

- Second, many of the names in BRENDA are difficult to parse. To help guide this, some of the ligands in BRENDA are associated with inchi values. This list can be downloaded by placing a blank query [here](https://www.brenda-enzymes.org/search_result.php?a=13). This can be saved and passed in to the `--brend-ligands` argument.

## Command to run for parsing: 

`python parse_brenda.py --brenda-flat-file data/raw/brenda_download.txt --brenda-ligands data/raw/brenda_ligands.csv --out-prefix results/out --no-standardize --load-prev --opsin-loc external_tools/opsin.jar`

## Analysis

After parsing the file, the output, `[out_name]_rxn_final.tsv` can be parsed for statistical analysis or downstream processing. Specifically, for this application, the script `analyze_num_rxns.py` provides a simple way to work with this:

- The function `preprocess` removes any entry that has Unknown entries in either its reactants or products 

- After this, all duplicate reactions for each EC number are removed, arbitrarily choosing to include any one of the data entries that has the duplicate reaction.

Sample way to run this script: 

`python analyze_num_rxns --parsed-data results/out_rxn_final.tsv`


## Pipeline

- EC classes and reactions are parsed from the flat file
- All ligands in the `brenda_ligands` supplement are extracted and included as alterantives.
- We try to resolve all brenda common names of substrates to smiles: 
        - As in `src/parse_brenda_compounds.csv` there are a number of resolvers and a number of name change edits. The name change edits are intended to change the query to something close that may be syntactically understood by the parser, like removing a haifen and replacing with whitespace. 
        - All resolvers are tried with all combinations of name substitutions. The name resolvers are: 
                - Using the provided inchi or chebi mapping from the `brenda_ligands` input
                - Pubchem bulk query api 
                - Opsin chemical parser
- After name resolution, we have an optional step to use MolVS to try to standardize compounds. This is time consuming and can be parallelized for efficiency. 
- The reaction (and compound) file is exported. 

Note: The program attempts to cache intermediates along the way and running with the same prefix and `--load-prev` arg will ensure that these steps are not repeated. 

## References

This package relies on the use of Opsin developed by Lowe et al., and their Java implementation is included in the folder "External tools". 

Lowe, Daniel M., et al. "Chemical name to structure: OPSIN, an open source solution." (2011): 739-753.


## TODO: 

- Add functionality for gene resolution from other repo
- Add export of kinetic reactions as well from other repo

# brenda-parser
A repository designated to parsing entries from the BRENDA database

## Installation

Conda requirements to run this parse can be found in environment.yml

## Downloads reqired:

This program relies on two different files being downloaded. 

- First, a recent copy of the brenda database can be downloaded at their website [here](https://www.brenda-enzymes.org/download_brenda_without_registration.php). This is the `brenda-flat-file` to include. This can be added to the data folder, `data/raw/brenda_download.txt`. 

- Second, many of the names in BRENDA are difficult to parse. To help guide this, some of the ligands in BRENDA are associated with inchi values. This list can be downloaded by placing a blank query [here](https://www.brenda-enzymes.org/search_result.php?a=13). This can be saved and passed in to the `--brend-ligands` argument.

## Command to run: 

`python parse_brenda.py --brenda-flat-file data/raw/brenda_download.txt --brenda-ligands data/raw/brenda_ligands.csv --out-prefix results/out --no-standardize --load-prev --opsin-loc external_tools/opsin.jar`


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


## TODO: 

- Add Esther's script and additional post processing to this script
- Delete the file `src/extract_gene_subs.py`, which is from an older version of this and only included as reference for ideas to post process
- Add this description to Esther's overleaf doc.
- Add parsed result files and inputs to dropbox for download and share with Esther

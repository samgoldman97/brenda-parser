# brenda_parse
A repository designated to parsing entries from the BRENDA database


Command to run: 

`python parse_brenda.py --brenda-flat-file data/raw/brenda_download.txt --brenda-ligands data/raw/brenda_ligands.csv --out-prefix results/out --no-standardize --load-prev --opsin-loc external_tools/opsin.jar`


TODO: 
- [done] Create conda env (export)
        - rdkit
        - requests
        - lxml
        - cirpy
        - molvs
        - seaborn
        - Command to export: 
                - `conda env export --name brenda_parse > enviornment.yml`
- Run full script 
- Add queries as specified by Esther
- Share repository
- [Down the line] Add in parsing for kinetics, etc. 






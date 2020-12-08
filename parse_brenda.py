"""Entry point into this code to parse the desired file. 

python parse_brenda.py --brenda-flat-file data/raw/BRENDA/brenda_download.txt --out-prefix results/BRENDA/parsed_brenda --brenda-ligands data/raw/BRENDA/brenda_ligands.csv --cirpy-log data/interim/BRENDA/cirpy_log.txt --load-prev --opsin-loc enzpred/data/brenda/opsin.jar

"""

from src.parse_brenda_main import main


if __name__=="__main__": 
    main()





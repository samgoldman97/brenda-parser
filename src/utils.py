"""Module containing some helper modules
"""

import os
import pickle
import json 
import numpy as np
import logging
from typing import Tuple, Optional, List, Any
import hashlib

import requests
import time
import shutil
import urllib.request as request
from contextlib import closing
from lxml import etree as ET
from collections import defaultdict

MAX_BATCH_SIZE = 199

def md5(key: str) -> str:
    """md5.

    Args:
        key (str): string to be hasehd
    Returns:
        Hashed encoding of str
    """
    return hashlib.md5(key.encode()).hexdigest()

def dump_json(obj: dict,
              outfile: str = "temp_dir/temp.json",
              pretty_print : bool =True) -> None:
    """pickle_obj.

    Helper fn to pickle object

    Args:
        obj (Any): dict
        outfile (str): outfile
        pretty_print (bool): If true, use tabs

    Returns: 
        None
    """
    if pretty_print:
        json.dump(obj, open(outfile, "w"), indent=2)
    else: 
        json.dump(obj, open(outfile, "w"))

def load_json(infile: str = "temp_dir/temp.p") -> Any:
    """load_json.

    Args:
        infile (str): infile, the name of input object

    Returns: 
        Any: the object loaded from pickled file

    """

    with open(infile, "r") as fp:
        return json.load(fp)

def pickle_obj(obj: Any, outfile: str = "temp_dir/temp.p") -> None:
    """pickle_obj.

    Helper fn to pickle object

    Args:
        obj (Any): obj
        outfile (str): outfile

    Returns: 
        None
    """
    with open(outfile, "wb") as fp:
        pickle.dump(obj, fp)

def pickle_load(infile: str = "temp_dir/temp.p") -> Any:
    """pickle_load.

    Args:
        infile (str): infile, the name of input object

    Returns: 
        Any: the object loaded from pickled file

    """
    with open(infile, "rb") as fp:
        return pickle.load(fp)

def make_dir(filename: str) -> None:
    """make_dir.

    Makes the directory that should contain this file

    Args:
        filename (str): filename

    Returns: 
        None
    """
    # Make outdir if it doesn't exist
    out_folder = os.path.dirname(filename)
    os.makedirs(out_folder, exist_ok=True)

def download_ftp(ftp_link: str, outfile: str):
    """download_ftp.

    Args:
        ftp_link (str): ftp_link
        outfile (str): outfile
    """

    with closing(request.urlopen(ftp_link)) as r:
        with open(outfile, 'wb') as f:
            shutil.copyfileobj(r, f)

def query_pubchem(ids: list,
                  query_type: str = "inchi",
                  save_file: str = "pubchem_save.txt",
                  rm_save: bool = True, 
                  encoding : str ="utf8",
                  return_single : bool = False) -> dict:
    """query_pubchem.

    Args:
        ids (list):
        query_type (str): inchi, chebi, or synonym
        save_file (str):
        rm_save (bool): If true, delete the file saved afterward
        encoding (str): Encoding to send request, defaults to utf-8
        return_single (bool): If true, only return the top hit for smiles

    Return:
        dict mapping ids to smiles lists
    """
    # Add options for query_type

    # 60 seconds
    WAIT_TIME = 60
    DOCHEADER = "<!DOCTYPE PCT-Data PUBLIC \"-//NCBI//NCBI PCTools/EN\" \"NCBI_PCTools.dtd\">"
    URL = "https://pubchem.ncbi.nlm.nih.gov/pug/pug.cgi"
    REQUIRED_HEADINGS = [
        "PCT-Data_input", "PCT-InputData", "PCT-InputData_query", "PCT-Query",
        "PCT-Query_type", "PCT-QueryType", "PCT-QueryType_id-exchange",
        "PCT-QueryIDExchange"
    ]

    QUERY_SUBTREE_NAMES = ["PCT-QueryIDExchange_input", "PCT-QueryUids"]
    OUTPUT_HEADERS = [
        "PCT-QueryIDExchange_operation-type", "PCT-QueryIDExchange_output-type",
        "PCT-QueryIDExchange_output-method", "PCT-QueryIDExchange_compression"
    ]
    OUTPUT_VALUES = ["same", "smiles", "file-pair", "none"]

    # Start building the query tree
    root = ET.Element("PCT-Data")
    cur_pos = root

    for new_heading in REQUIRED_HEADINGS:
        cur_pos = ET.SubElement(cur_pos, new_heading)

    # Navigate down to where we add the inchis
    query_subtree = cur_pos
    for query_subtree_name in QUERY_SUBTREE_NAMES:
        query_subtree = ET.SubElement(query_subtree, query_subtree_name)

    # Now add the things SPECIFIC to inchi
    if query_type == "inchi":
        query_root, query_name = "PCT-QueryUids_inchis", "PCT-QueryUids_inchis_E"
        query_subtree = ET.SubElement(query_subtree, query_root)
        for id_ in ids:
            new_id = ET.SubElement(query_subtree, query_name)
            # give this the id text
            try:
                new_id.text = id_
            except ValueError:
                logging.warning(f"Couldn't query {id_} due to bad encoding")

    elif query_type == "synonym":
        query_root, query_name = "PCT-QueryUids_synonyms", "PCT-QueryUids_synonyms_E"
        query_subtree = ET.SubElement(query_subtree, query_root)
        for id_ in ids:
            new_id = ET.SubElement(query_subtree, query_name)
            # give this the id text
            try:
                new_id.text = id_
            except ValueError:
                logging.warning(f"Couldn't query {id_} due to bad encoding")

    elif query_type == "chebi":
        for i in ["PCT-QueryUids_source-ids", "PCT-RegistryIDs"]:
            query_subtree = ET.SubElement(query_subtree, i)
        source_id_name = ET.SubElement(query_subtree,
                                       "PCT-RegistryIDs_source-name")
        source_id_name.text = "ChEBI"

        query_subtree = ET.SubElement(query_subtree,
                                      "PCT-RegistryIDs_source-ids")
        for id_ in ids:
            new_id = ET.SubElement(query_subtree,
                                   "PCT-RegistryIDs_source-ids_E")
            # give this the id text
            try:
                new_id.text = id_
            except ValueError:
                logging.warning(f"Couldn't query {id_} due to bad encoding")
    else:
        raise NotImplemented

    # Go back up to to current position holder
    # Add the output specification
    for output_header, output_value in zip(OUTPUT_HEADERS, OUTPUT_VALUES):
        output_xml = ET.SubElement(cur_pos, output_header)
        output_xml.set("value", output_value)

    out_xml = ET.tostring(root,
                          encoding=encoding,
                          method="xml",
                          xml_declaration=True,
                          doctype=DOCHEADER).decode()

    # Post the request!
    resp = requests.post(URL, data=out_xml.encode('utf-8'))

    # Handle response and build a request to check on status
    resp_tree = ET.fromstring(resp.text)
    waiting_id = resp_tree.xpath("//PCT-Waiting_reqid")
    waiting_id = waiting_id[0].text if waiting_id else None

    STATUS_CHECK_HEADERS = [
        "PCT-Data_input", "PCT-InputData", "PCT-InputData_request",
        "PCT-Request"
    ]

    root = ET.Element("PCT-Data")
    cur_pos = root
    for header in STATUS_CHECK_HEADERS:
        cur_pos = ET.SubElement(cur_pos, header)
    req_id = ET.SubElement(cur_pos, "PCT-Request_reqid")
    req_id.text = waiting_id
    req_type = ET.SubElement(cur_pos, "PCT-Request_type")
    req_type.set("value", "status")
    query_xml = ET.tostring(root,
                            encoding=encoding,
                            method="xml",
                            xml_declaration=True,
                            doctype=DOCHEADER).decode()

    download_link = None
    waiting_time = 0

    # TODO: Add stop timeout condition?
    # Repeatedly query to see if the results are done, then sleep for WAITIME
    # in case they aren't
    while not download_link:
        resp = requests.post(URL, data=query_xml.encode('utf-8'))
        resp_tree = ET.fromstring(resp.text)
        download_link = resp_tree.xpath("//PCT-Download-URL_url")
        download_link = download_link[0].text if download_link else None
        time.sleep(WAIT_TIME)
        waiting_time += WAIT_TIME
        logging.warning(f"Waiting time: {waiting_time} seconds")

    # At conclusion, download the ftp file
    download_ftp(download_link, save_file)

    # Also parse this
    ret_dict = defaultdict(lambda: [])
    with open(save_file, "r") as fp:
        for linenum, line in enumerate(fp):
            line = line.strip()
            split_line = line.split("\t")
            if len(split_line) == 2:
                mol, smiles = split_line
                mol = mol.strip()
                smiles = smiles.strip()
                ret_dict[mol].append(smiles)
            else:
                logging.debug(f"No smiles mol found for {line}")

    # If we should only return a single item and not a list
    if return_single: 
        ret_dict = {k : v[0] for k,v in ret_dict.items() if len(v) > 0}

    # Remove temp file
    if os.path.exists(save_file) and rm_save:
        os.remove(save_file)

    return ret_dict

def debug_pubchem():
    """ Helper fn for debugging pubchem"""
    test_inchis = [
        "InChI=1S/I2/c1-2", "InChI=1S/C30H46O3/",
        "InChI=1S/C6H11NO2/c8-6(9)5-3-1-2-4-7-5/h5,7H,1-4H2,(H,8,9)/t5-/m0/s1"
    ]
    test_chebis = ["chebi:61185"]
    query_syns = ["glucose", "NADH"]

    mapping = query_pubchem(test_inchis,
                            query_type="inchi",
                            save_file="pubchem_inchi.txt")
    print(mapping)
    mapping = query_pubchem(test_chebis,
                            query_type="chebi",
                            save_file="pubchem_chebi.txt")
    mapping = query_pubchem(query_syns,
                            query_type="synonym",
                            save_file="pubchem_syns.txt")


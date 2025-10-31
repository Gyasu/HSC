#!/usr/bin/env python3

import csv
import gzip
import json
import os
import logging
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# from biopython
from Bio import SeqIO
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder


from hsc.utils import seq_utils, pdb_utils

warnings.simplefilter('ignore', BiopythonWarning)


def parse_cmd():
    """
    Parse command-line arguments for computing HSC scores.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including config file, input file, radius, database,
        overwrite flag, verbosity, and log file.
    """
    parser = ArgumentParser()
    
    parser.add_argument(
        '-c', '--config', dest='config', required=True,
        type=str, help='A JSON file specifying options.'
    )
    parser.add_argument(
        '-i', '--input', dest='input', required=True,
        type=str, help='Input file specifying the UniProt ID and PDB file, e.g., "P13929 AF-P13929-F1-model_v4.pdb".'
    ) 
    parser.add_argument(
        '-w', '--overwrite', dest='overwrite', required=False,
        action='store_true', help='Whether to overwrite already computed HSC scores.'
    )
    parser.add_argument(
        '-r', '--radius', dest='radius', type=float, default=8,
        help='Radius within which to include sites.'
    )
    parser.add_argument(
        '-v', '--verbose', dest='verbose', required=False,
        action='store_true', help='Whether to output verbose data, including number of contacting residues and variants in the neighborhood.'
    )
    parser.add_argument(
        '-l', '--log', dest='log', default='hsc.log',
        help='The file to which to write detailed computing logs.'
    )
    
    return parser.parse_args()

def get_ensembl_accession(record):
    """

    Parameters
    ----------
    record

    Returns
    -------

    """
    parts = record.id.split('.')
    return parts[0]

def get_ccds_accession(record):
    """

    Parameters
    ----------
    record

    Returns
    -------

    """
    parts = record.id.split('|')
    return parts[0]

def get_uniprot_accession(record):
    """

    Parameters
    ----------
    record

    Returns
    -------

    """
    parts = record.id.split('|')
    return parts[1]


def get_pdb_chain(pdb_file, pdb_chain):
    """
    Creates a Bio.PDB.Chain object for the requested PDB chain.

    Parameters
    ----------
    pdb_id : str
        Identifier of the PDB chain as a five-letter string.
    pdb_chain : str
        Identifier of the PDB chain as a five-letter string.
    pdb_db : str
        Path to the local PDB database.

    Returns
    -------
    Bio.PDB.Chain
        The requested PDB chain as a Bio.PDB.Chain object.

    """
    # read in the PDB file
    pdb_parser = PDBParser(PERMISSIVE=1)
    try:
        if pdb_file.endswith('.gz'):
            with gzip.open(pdb_file, 'rt') as file:
                structure = pdb_parser.get_structure(id="given_pdb", file=file)
        else:
            structure = pdb_parser.get_structure(id="given_pdb", file=pdb_file)
    except (FileNotFoundError, ValueError, IOError) as e:
        print('PDB file cannot be retrieved:', pdb_file)
        return None
    
    try:
        chain = structure[0][pdb_chain]
    except KeyError:
        print('No chain ' + pdb_chain + ' was found in ' + pdb_file)
        return None
    return chain


def parse_config(config):
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    with open(config, 'rt') as ipf:
        configs = json.load(ipf)

    # do necessary sanity checks before return
    return configs


def count_variants(variants):
    """
    Collects the statistics about position-specific counts of missense and
    synonymous variants.

    Parameters
    ----------
    variants : list
        A list of variant identifiers: ['A123B', 'C456D']

    Returns
    -------
    dict
        A dictionary where the key is amino acid position and the value is
        the number of variants at this position. One dictionary for missense
        variants and one dictionary for synonymous variants.

    """

    missense_counts = defaultdict(int)
    synonymous_counts = defaultdict(int)
    for variant in variants:
        vv = variant[0]
        count = int(variant[1])
        AN = int(variant[2])

        MAF = (count / AN) 

        if MAF > 0.5:
            result = 1 - MAF
        else:
            result = MAF
                
        w = vv[0]  # wild-type amino acid
        v = vv[-1]  # mutant amino acid
        pos = vv[1:-1]  # position in the protein sequence
        

        if w != v:  # missense variant
            missense_counts[int(pos)] += result
        else:  # synonymous variant
            synonymous_counts[int(pos)] += result
    return missense_counts, synonymous_counts

class NoValidCDSError(ValueError):
    pass

class NoTranscriptRecordError(KeyError):
    pass

class NoRecordInDictionaryError(KeyError):
    pass

class NoVariantsError(KeyError):
    pass

def retrieve_data(pep_seq, uniprot_id, enst_ids, cds_dict, variant_dict):
    """
    Retrieve the coding sequence (CDS) and variant information for a given 
    UniProt ID, matching it with valid Ensembl transcript(s).

    Parameters
    ----------
    pep_seq : str
        Protein sequence corresponding to the UniProt ID.
    uniprot_id : str
        UniProt identifier.
    enst_ids : list of str
        List of Ensembl transcript IDs associated with the UniProt ID.
    cds_dict : dict
        Dictionary mapping Ensembl transcript IDs to CDS SeqRecords.
    variant_dict : dict
        Dictionary mapping Ensembl transcript IDs to variant data.

    Returns
    -------
    enst_id : str
        Selected Ensembl transcript ID.
    cds_seq : str
        Coding DNA sequence corresponding to the selected transcript.
    variants : list or dict
        Variant data associated with the selected transcript.
    """

    # Filter valid transcripts
    valid_transcripts = []
    for enst_id in enst_ids:
        cds_seq = cds_dict[enst_id].seq
        if not seq_utils.is_valid_cds(cds_seq):
            print(f"Warning: Invalid CDS for transcript {enst_id}. Skipping.")
            continue
        # Check if protein and CDS length are compatible
        if len(pep_seq) == len(cds_seq) // 3 - 1:
            valid_transcripts.append(enst_id)

    if not valid_transcripts:
        raise ValueError(f"Error: No valid transcripts compatible with UniProt ID {uniprot_id}: {enst_ids}")

    # If only one valid transcript, return it
    if len(valid_transcripts) == 1:
        enst_id = valid_transcripts[0]
        cds_seq = cds_dict[enst_id].seq
        if enst_id not in variant_dict:
            raise KeyError(f"Error: No variant record for {uniprot_id} in gnomAD.")
        variants = variant_dict[enst_id]['variants']
        return enst_id, cds_seq, variants

    # If multiple valid transcripts, select the one with the most variant positions
    selected_transcript = None
    max_variants = -1
    for enst_id in valid_transcripts:
        num_variants = len(variant_dict.get(enst_id, {}).get('variants', []))
        if num_variants > max_variants:
            max_variants = num_variants
            selected_transcript = enst_id

    if selected_transcript is None:
        raise KeyError(f"Error: None of the valid transcripts have variant data for UniProt ID {uniprot_id}.")

    cds_seq = cds_dict[selected_transcript].seq
    variants = variant_dict[selected_transcript]['variants']

    return selected_transcript, cds_seq, variants


def get_dataset_headers():
    """
    Returns column name for each feature of the dataset. Every time a new
    features is added, this function needs to be updated.

    Returns
    -------

    """
    header = [
        'uniprot_id', 'enst_id', 'uniprot_pos', 'uniprot_aa',
        'seq_separations', 'num_contacts', 'syn_var_sites',
        'total_syn_sites', 'mis_var_sites', 'total_mis_sites',
        'cs_syn_poss', 'cs_mis_poss', 'cs_gc_content', 'cs_syn_prob',
        'cs_syn_obs', 'cs_mis_prob', 'log_cs_mis_obs', 'log_mis_pmt_mean', 'log_mis_pmt_sd','HSCZ',
        'mis_p_value', 'syn_pmt_mean', 'syn_pmt_sd', 'syn_p_value',
        'enst_syn_obs', 'enst_mis_obs', 'enst_syn_exp', 'enst_mis_exp', 
        'plddt', 'uniprot_length'
    ]
    return header


def load_datasets(configs):
    """
    Load all required datasets for computing HSC scores.

    Parameters
    ----------
    configs : dict
        Dictionary containing file paths for the required datasets:
        - 'ensembl_cds': path to ENSEMBL CDS fasta (gzipped)
        - 'uniprot_pep': path to UniProt peptide fasta (gzipped)
        - 'gnomad_variants': path to gnomAD transcript-level variants (JSON)
        - 'uniprot_to_enst': path to UniProt-to-Ensembl mapping file (JSON)
        - 'enst_mp_counts': path to transcript mutation probability and counts

    Returns
    -------
    tuple
        (enst_cds_dict, pep_dict, enst_variants, uniprot_to_enst, enst_mp_counts)
    """
    
    # Load ENSEMBL CDS sequences
    print('Loading ENSEMBL CDS sequences...')
    with gzip.open(configs['ensembl_cds'], 'rt') as cds_handle:
        enst_cds_dict = SeqIO.to_dict(
            SeqIO.parse(cds_handle, format='fasta'),
            key_function=get_ensembl_accession
        )

    # Load gnomAD transcript-level variants
    print('Loading gnomAD variant database...')
    with open(configs['gnomad_variants'], 'rt') as variant_handle:
        enst_variants = json.load(variant_handle)

    # Load UniProt-to-Ensembl transcript mapping
    print('Loading UniProt to ENST mapping...')
    with open(configs['uniprot_to_enst'], 'rt') as map_handle:
        uniprot_to_enst = json.load(map_handle)

    # Load transcript mutation probabilities and variant counts
    print('Loading transcript mutation probabilities and variant counts...')
    enst_mp_counts = seq_utils.read_enst_mp_count_dist(configs['enst_mp_counts'])

    return enst_cds_dict, enst_variants, uniprot_to_enst, enst_mp_counts

def get_pep_seq_from_pdb(pdb_file, pdb_chain):
    if pdb_file.endswith('.gz'):
        with gzip.open(pdb_file, 'rt') as f:
            structure = PDBParser().get_structure('protein', f)
    else:
        structure = PDBParser().get_structure('protein', pdb_file)
    
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        if pp.get_sequence():
            return pp.get_sequence()
    
    return None

def main():
    
    """
        Main pipeline for computing Human Site Constraint (HSC) scores.

        Steps:
        1. Parse arguments and configuration.
        2. Load CDS, variant, and transcript mapping datasets.
        3. For each UniProt–AlphaFold pair, compute per-residue constraint statistics.
        4. Save results as a tab-separated file.

    Returns
    -------
        None
            Writes <uniprot_id>_hsc.tsv files to the configured output directory.
    """

    args = parse_cmd()
    configs = parse_config(args.config)

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        filemode='w',
        format='%(levelname)s:%(asctime)s:%(message)s'
    )

    logging.info("Starting HSC computation pipeline...")

    output_dir = os.path.abspath(configs['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    
    # Load datasets
    cds_dict, variant_dict, uniprot_to_enst, enst_mp_counts = load_datasets(configs)
    
    with open(args.input, 'rt') as mapping_file:
        uniprot_pdb_pairs = [line.strip().split() for line in mapping_file]
    
    # Error counters
    error_counts = {
        "no_enst_for_uniprot": 0,
        "no_valid_cds": 0,
        "no_variant_data": 0,
        "bad_structure": 0,
        "no_enst_mp": 0,
    }

    # Compute HSC scores for each UniProt-PDB pair
    for uniprot_id, pdb_model_name in uniprot_pdb_pairs:
        logging.info(f"Processing {uniprot_id} with model {pdb_model_name}")

        # Determine PDB file path and chain
        pdb_file = os.path.join(configs['pdb_dir'], pdb_model_name) + '.gz'
        pdb_chain = 'A'
        pep_seq = get_pep_seq_from_pdb(pdb_file, pdb_chain)

        # Skip computation if output already exists and overwrite is not set
        output_file = os.path.join(output_dir, f"{uniprot_id}_hsc.tsv")
        if os.path.exists(output_file) and not args.overwrite:
            logging.info(f"{uniprot_id}_hsc.tsv already exists. Skipping.")
            continue

        hsc_scores = []

        try:
            enst_ids = uniprot_to_enst[uniprot_id]
        except KeyError:
            logging.error(f"No ENST transcript IDs were mapped to {uniprot_id}") 
            error_counts['no_enst_for_uniprot'] += 1
            continue

        try:
            right_enst, cds, variants = retrieve_data(
                pep_seq, uniprot_id, enst_ids, cds_dict, variant_dict
            )
        except ValueError:
            logging.error(f"No valid CDS found for {uniprot_id}")
            error_counts['no_valid_cds'] += 1
            continue

        except KeyError:
            logging.error(f"No transcript record found for {uniprot_id} in gnomAD.")
            error_counts['no_variant_data'] += 1
            continue

        logging.info(f"Computing HSC features for: {uniprot_id}, {right_enst}, {pdb_file}")



        chain = get_pdb_chain(pdb_file, pdb_chain)
        if chain is None:
            logging.warning(f"Missing chain {pdb_chain} in structure {pdb_file}. Skipping.")
            error_counts["bad_structure"] += 1
            continue

        
        all_aa_residues = [aa for aa in chain.get_residues() if is_aa(aa, standard=True)]
            
        if not all_aa_residues or len(all_aa_residues) / len(pep_seq) < 1.0 / 3.0:
            logging.warning(f"Low-quality structure for {uniprot_id} ({pdb_model_name}). Skipping.")
            error_counts["bad_structure"] += 1
            continue
        
        
        all_contacts = pdb_utils.search_for_all_contacts(all_aa_residues, radius=args.radius)

        indexed_contacts = defaultdict(list)
        for c in all_contacts:
            indexed_contacts[c.get_res_a()].append(c.get_res_b())
            indexed_contacts[c.get_res_b()].append(c.get_res_a())


        cds = cds[:-3]  # remove the stop codon
        codon_mutation_rates = seq_utils.get_codon_mutation_rates(cds)
        all_cds_ns_counts = seq_utils.count_poss_ns_variants(cds)
        cds_ns_sites = seq_utils.count_ns_sites(cds)

        if len(codon_mutation_rates) < len(all_aa_residues):
            logging.warning(f"Residue–sequence mismatch for {uniprot_id}. Skipping.")
            error_counts["bad_structure"] += 1
            continue

        mis_counts, syn_counts = count_variants(variants)
        site_var_mis = {pos: 1 for pos in mis_counts}
        site_var_syn = {pos: 1 for pos in syn_counts}

        try:
            total_mis_exp = enst_mp_counts[right_enst][-3]
            total_syn_exp = enst_mp_counts[right_enst][-4]
            mis_dist = enst_mp_counts[right_enst][-1]
            syn_dist = enst_mp_counts[right_enst][-2]
        except KeyError:
            logging.error(f"Missing ENST record in counts: {right_enst}")
            error_counts["no_enst_mp"] += 1
            continue
        
        try:
            codon_mis_probs = [x[1] for x in codon_mutation_rates]
            codon_syn_probs = [x[0] for x in codon_mutation_rates]
            mis_pmt_matrix = seq_utils.permute_variants_dist(total_mis_exp, len(pep_seq), codon_mis_probs, syn_dist)
            syn_pmt_matrix = seq_utils.permute_variants_dist(total_syn_exp, len(pep_seq), codon_syn_probs, syn_dist)
        

        except ValueError:
            logging.error(f'Protein Length Mismatch')
            continue

        valid_case = True
        for seq_pos, seq_aa in enumerate(pep_seq, start=1):
            try:
                res = chain[seq_pos]
            except KeyError:
                continue
            
            if seq1(res.get_resname()) != seq_aa:
                logging.warning(f"Residue mismatch at {seq_pos} in {uniprot_id}. Skipping protein.")
                valid_case = False
                break
                
            plddt = res['CA'].get_bfactor()

            contact_res = indexed_contacts[res]
            num_contacts = len(contact_res)
            contacts_pdb_pos = [r.get_id()[1] for r in contact_res]
            seq_seps = ';'.join(
                str(x) for x in [i - seq_pos for i in contacts_pdb_pos]
            )

            mis_var_sites = site_var_mis.setdefault(seq_pos, 0)
            total_mis_sites = cds_ns_sites[seq_pos - 1][0]
            syn_var_sites = site_var_syn.setdefault(seq_pos, 0)
            total_syn_sites = cds_ns_sites[seq_pos - 1][1]
            total_missense_obs = mis_counts.setdefault(seq_pos, 0)
            total_synonymous_obs = syn_counts.setdefault(seq_pos, 0)
            total_missense_poss = all_cds_ns_counts[seq_pos - 1][0]
            total_synonyms_poss = all_cds_ns_counts[seq_pos - 1][1]
            total_synonymous_rate = codon_mutation_rates[seq_pos - 1][0]
            total_missense_rate = codon_mutation_rates[seq_pos - 1][1]
            for j in contacts_pdb_pos:
                # count the total # observed variants in contacting residues
                mis_var_sites += site_var_mis.setdefault(j, 0)
                syn_var_sites += site_var_syn.setdefault(j, 0)
                total_missense_obs += mis_counts.setdefault(j, 0)
                total_synonymous_obs += syn_counts.setdefault(j, 0)

                # count the total # expected variants
                try:
                    total_missense_poss += all_cds_ns_counts[j - 1][0]
                    total_synonyms_poss += all_cds_ns_counts[j - 1][1]
                    total_synonymous_rate += codon_mutation_rates[j - 1][0]
                    total_missense_rate += codon_mutation_rates[j - 1][1]
                    total_mis_sites += cds_ns_sites[j - 1][0]
                    total_syn_sites += cds_ns_sites[j - 1][1]
                except IndexError:
                    valid_case = False
                    break
            if not valid_case:
                break

            try:
                seq_context = seq_utils.get_codon_seq_context(
                    contacts_pdb_pos + [seq_pos], cds
                )
            except IndexError:
                break

            # compute the GC content of the sequence context
            if len(seq_context) == 0:
                print('No nucleotides were found in sequence context!')
                continue
            gc_fraction = seq_utils.gc_content(seq_context)

            total_missense_obs = np.log10(total_missense_obs + 1e-12)
            total_synonymous_obs = np.log10(total_synonymous_obs + 1e-12)

            mis_pmt_mean, mis_pmt_sd, mis_p_value = seq_utils.get_permutation_stats(
                mis_pmt_matrix, contacts_pdb_pos + [seq_pos], total_missense_obs
            )
            syn_pmt_mean, syn_pmt_sd, syn_p_value = seq_utils.get_permutation_stats(
                syn_pmt_matrix, contacts_pdb_pos + [seq_pos], total_synonymous_obs
            )

            # compute the fraction of expected missense variants
            hsc_scores.append(
                [
                    uniprot_id, right_enst, seq_pos, seq_aa, seq_seps,
                    num_contacts + 1,
                    syn_var_sites,
                    '{:.3f}'.format(total_syn_sites),
                    mis_var_sites,
                    '{:.3f}'.format(total_mis_sites),
                    total_synonyms_poss,
                    total_missense_poss,
                    '{:.3f}'.format(gc_fraction),
                    '{:.3e}'.format(total_synonymous_rate),
                    total_synonymous_obs,
                    '{:.3e}'.format(total_missense_rate),
                    total_missense_obs,
                    mis_pmt_mean,
                    mis_pmt_sd,
                    '{:.3f}'.format((total_missense_obs - mis_pmt_mean) / mis_pmt_sd),  
                    '{:.3e}'.format(mis_p_value),
                    '{:.3f}'.format(syn_pmt_mean),
                    '{:.3f}'.format(syn_pmt_sd),
                    '{:.3e}'.format(syn_p_value),
                    enst_mp_counts[right_enst][2],
                    enst_mp_counts[right_enst][4],
                    total_syn_exp,
                    total_mis_exp,
                    '{:.2f}'.format(plddt),
                    len(pep_seq)
                ]
            )

        if not valid_case:
            continue

        with open(output_file, "wt") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            writer.writerow(get_dataset_headers())
            writer.writerows(hsc_scores)

        logging.info(f"Successfully computed HSC for {uniprot_id}.")



    # print total errors
    logging.info("Pipeline completed.")
    for k, v in error_counts.items():
        logging.info(f"{k}: {v}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Example command-line usage:
#python constraintometer/main.py -c config.json -r 8 -i inputs/weighted_cosmis_input_1.txt -d AlphaFold -l test_log.log


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
from Bio import SeqIO
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder


from constraintometer.utils import seq_utils, pdb_utils

warnings.simplefilter('ignore', BiopythonWarning)


def parse_cmd():
    """

    Returns
    -------

    """
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True,
                        type=str, help='A JSON file specifying options.')
    parser.add_argument('-i', '--input', dest='input', required=True,
                        type=str, help='''Ensembl transcript ID of the 
                        transcript for which to compute a MTR3D profile.''')
    parser.add_argument('-w', '--overwrite', dest='overwrite', required=False,
                        action='store_true', help='''Whether to overwrite 
                        already computed MTR3D scores.''')
    parser.add_argument('-r', '--radius', dest='radius', type=float, default=8,
                        help='''Radius within which to include sites.''')
    parser.add_argument('-d', '--database', dest='database', required=True,
                       default='SWISS-MODEL', help='Structure database to be used.')
    parser.add_argument('-v', '--verbose', dest='verbose', required=False,
                        action='store_true', help='''Whether to output verbose
                        data: number of contacting residues and number of 
                        missense and synonymous variants in the neighborhood
                        of the mutation site.''')
    parser.add_argument('-l', '--log', dest='log', default='cosmis.log',
                        help='''The file to which to write detailed computing logs.''')
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
    #
    missense_counts = defaultdict(int)
    synonymous_counts = defaultdict(int)
    for variant in variants:
        vv = variant[0]
        count = int(variant[1])
        AN = int(variant[2])

        MAF = (count / AN) #* 100

        if MAF > 0.5:
            result = 1 - MAF
        else:
            result = MAF
        # result = 1 / (2 * MAF * (1 - MAF))**(-1)
        

        # print(f'MAF is {MAF} and result is {result}')
        
        w = vv[0]  # wild-type amino acid
        v = vv[-1]  # mutant amino acid
        pos = vv[1:-1]  # position in the protein sequence
        
        # only consider rare variants
        # if int(ac) / int(an) > 0.001: 
        #    continue
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

def retrieve_data(uniprot_id, enst_ids, pep_dict, cds_dict, variant_dict):
    """
    """
    pep_seq = pep_dict[uniprot_id]

    valid_ensts = []
    for enst_id in enst_ids:
        cds_seq = cds_dict[enst_id].seq
        # skip if the CDS is incomplete
        if not seq_utils.is_valid_cds(cds_seq):
            print('Error: Invalid CDS.'.format(enst_id))
            continue
        if len(pep_seq) == len(cds_seq) // 3 - 1:
            valid_ensts.append(enst_id)
    if not valid_ensts:
        raise ValueError(
            'Error: {} are not compatible with {}.'.format(enst_ids, uniprot_id)
        )

    if len(valid_ensts) == 1:
        enst_id = valid_ensts[0]
        cds = cds_dict[enst_id].seq
        if enst_id not in variant_dict.keys():
           raise KeyError('Error: No record for {} in gnomAD.'.format(uniprot_id))
        variants = variant_dict[enst_id]['variants']
        return enst_id, pep_seq, cds, variants

    # if multiple transcripts are valid
    # get the one with most variable positions
    max_len = 0
    right_enst = ''
    for enst_id in valid_ensts:
        try:
            var_pos = len(variant_dict[enst_id]['variants'])
        except KeyError:
            continue
        if max_len < var_pos:
            max_len = var_pos
            right_enst = enst_id
    cds = cds_dict[right_enst].seq
    variants = variant_dict[right_enst]['variants']

    return right_enst, pep_seq, cds, variants



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
        'cs_syn_obs', 'cs_mis_prob', 'cs_mis_obs', 'mis_pmt_mean', 'mis_pmt_sd','cosmis',
        'mis_p_value', 'syn_pmt_mean', 'syn_pmt_sd', 'syn_p_value',
        'enst_syn_obs', 'enst_mis_obs', 'enst_syn_exp', 'enst_mis_exp', 
        'plddt', 'uniprot_length'
    ]
    return header


def load_datasets(configs):
    """

    Parameters
    ----------
    configs

    Returns
    -------

    """
    # ENSEMBL cds
    print('Reading ENSEMBL CDS database ...')
    with gzip.open(configs['ensembl_cds'], 'rt') as cds_handle:
        enst_cds_dict = SeqIO.to_dict(
            SeqIO.parse(cds_handle, format='fasta'),
            key_function=get_ensembl_accession
        )
        
    # ENSEMBL peptide sequences
    print('Reading UniProt protein sequence database ...')
    with gzip.open(configs['uniprot_pep'], 'rt') as pep_handle:
        pep_dict = SeqIO.to_dict(
            SeqIO.parse(pep_handle, format='fasta'),
            key_function=get_uniprot_accession
        )
    
    # Parse gnomad transcript-level variants
    print('Reading gnomAD variant database ...')
    with open(configs['gnomad_variants'], 'rt') as variant_handle:
        # transcript_variants will be a dict of dicts where major version
        # ENSEMBL transcript IDs are the first level keys and "ccds", "ensp",
        # "swissprot", "variants" are the second level keys. The value of each
        # second-level key is a Python list.
        enst_variants = json.load(variant_handle)
    '''
    print('Reading RGC variant database ...')
    with open(configs['RGC_variants'], 'rt') as RGC_variant_handle:
        # transcript_variants will be a dict of dicts where major version
        # ENSEMBL transcript IDs are the first level keys and "ccds", "ensp",
        # "swissprot", "variants" are the second level keys. The value of each
        # second-level key is a Python list.
        enst_variants = json.load(RGC_variant_handle)
    '''
    # parse the file that maps Ensembl transcript IDs to PDB IDs
    print('Reading uniprot to ENST mapping file ...')
    with open(configs['uniprot_to_enst'], 'rt') as ipf:
        uniprot_to_enst = json.load(ipf)


    # get transcript mutation probabilities and variant counts
    print('Reading transcript mutation probabilities and variant counts ...')
    enst_mp_counts = seq_utils.read_enst_mp_count_dist(configs['enst_mp_counts'])

    return (enst_cds_dict, pep_dict, enst_variants, uniprot_to_enst, enst_mp_counts)

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

    Returns
    -------

    """
    # Parse command-line arguments
    args = parse_cmd()

    # Configure the logging system
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        filemode='w',
        format='%(levelname)s:%(asctime)s:%(message)s'
    )

    # Parse configuration file
    configs = parse_config(args.config)

    # Load datasets
    cds_dict, pep_dict, variant_dict, uniprot_to_enst, enst_mp_counts = load_datasets(configs)

    # directory where to store the output files
    output_dir = os.path.abspath(configs['output_dir'])

    # read UniProt ID to swiss model mapping
    with open(args.input, 'rt') as ipf:
        uniprot_sm_mapping = [line.strip().split() for line in ipf]

    # compute constraint scores
    for uniprot_id, model_path in uniprot_sm_mapping:
        logging.info(f"Checking for {uniprot_id} and {model_path}")
        print(uniprot_id)
        if args.database == 'SWISS-MODEL':
            pdb_file = os.path.join(configs['pdb_dir'], model_path)
            pdb_chain = model_path.split('.')[-2]
        else:
            pdb_file = os.path.join(configs['pdb_dir'], model_path) + '.gz'
            pdb_chain = 'A'
            pep_seq = get_pep_seq_from_pdb(pdb_file, pdb_chain)
        if os.path.exists(
                os.path.join(output_dir, uniprot_id + '_cosmis.tsv')
        ) and not args.overwrite:
            logging.info(f"{uniprot_id}_cosmis.tsv already exists. Skipped.")
            continue
        
        cosmis = []

        try:
            enst_ids = uniprot_to_enst[uniprot_id]
        except KeyError:
            logging.critical(
                'No ENST transcript IDs were mapped to {}.'.format(uniprot_id)
            )
            continue

        try:
            right_enst, pep_seq, cds, variants = retrieve_data(
                uniprot_id, enst_ids, pep_dict, cds_dict, variant_dict
            )
        except ValueError:
            logging.critical('No valid CDS found for {}.'.format(uniprot_id))
            continue
        except KeyError:
            logging.critical('No transcript record found for {} in gnomAD.'.format(uniprot_id))
            continue

        # print message
        logging.info(f"Computing COSMIS features for: {uniprot_id}, {right_enst}, {pdb_file}")

        chain = get_pdb_chain(pdb_file, pdb_chain)

        if chain is None:
            print(
                'ERROR: %s not found in structure: %s!' % (pdb_chain, pdb_file))
            print('Skip to the next protein ...')
            continue

        if args.database == 'AlphaFold':
            # exclude residues that are of very low confidence, i.e. pLDDT < 50
            all_aa_residues = [
                aa for aa in chain.get_residues()
                if is_aa(aa, standard=True)
            ]
            if len(all_aa_residues) / len(pep_seq) < 1.0 / 3.0:
                logging.critical(
                    'Confident residues in AlphaFold2 model cover less than '
                    'one third of the peptide sequence: {} {}.'.format(uniprot_id, pdb_file)
                )
                continue
        else:
            all_aa_residues = [aa for aa in chain.get_residues() if is_aa(aa, standard=True)]
        if not all_aa_residues:
            logging.critical(
                'No confident residues found in the given structure'
                '{} for {}.'.format(pdb_file, uniprot_id)
            )
            continue
        all_contacts = pdb_utils.search_for_all_contacts(
            all_aa_residues, radius=args.radius
        )

        # calculate expected counts for each codon
        cds = cds[:-3]  # remove the stop codon
        codon_mutation_rates = seq_utils.get_codon_mutation_rates(cds)
        all_cds_ns_counts = seq_utils.count_poss_ns_variants(cds)
        cds_ns_sites = seq_utils.count_ns_sites(cds)

        if len(codon_mutation_rates) < len(all_aa_residues):
            print('ERROR: peptide sequence has less residues than structure!')
            print('Skip to the next protein ...')
            continue

        # tabulate variants at each site
        # missense_counts and synonymous_counts are dictionary that maps
        # amino acid positions to variant counts
        missense_counts, synonymous_counts = count_variants(variants)


        # convert variant count to site variability
        site_variability_missense = {
            pos: 1 for pos, _ in missense_counts.items()
        }
        site_variability_synonymous = {
            pos: 1 for pos, _ in synonymous_counts.items()
        }

        # compute the total number of missense variants
        try:
            total_exp_mis_counts = enst_mp_counts[right_enst][-3]
            total_exp_syn_counts = enst_mp_counts[right_enst][-4]
            mis_dist = enst_mp_counts[right_enst][-1]
            syn_dist = enst_mp_counts[right_enst][-2]
          
            # logging.info(f'this is mis_dist {mis_dist}')
        except KeyError:
            logging.critical('Transcript {} not found in {}'.format(right_enst, configs[
                'enst_mp_counts']))        

        # permutation test
        codon_mis_probs = [x[1] for x in codon_mutation_rates]
        codon_syn_probs = [x[0] for x in codon_mutation_rates]
        mis_p = codon_mis_probs / np.sum(codon_mis_probs)
        syn_p = codon_syn_probs / np.sum(codon_syn_probs)
        try:
            mis_pmt_matrix = seq_utils.permute_variants_dist(
                total_exp_mis_counts, len(pep_seq), mis_p, syn_dist
            )
            syn_pmt_matrix = seq_utils.permute_variants_dist(
                total_exp_syn_counts, len(pep_seq), syn_p, syn_dist
            )
        
        except ValueError:
            logging.critical('Protein Length Mismatch')
            continue

        # index all contacts by residue ID
        indexed_contacts = defaultdict(list)
        for c in all_contacts:
            indexed_contacts[c.get_res_a()].append(c.get_res_b())
            indexed_contacts[c.get_res_b()].append(c.get_res_a())

        valid_case = True
        for seq_pos, seq_aa in enumerate(pep_seq, start=1):
            try:
                res = chain[seq_pos]
            except KeyError:
                print('PDB file is missing residue:', seq_aa, 'at', seq_pos)
                continue
            pdb_aa = seq1(res.get_resname())
            if seq_aa != pdb_aa:
                print('Residue in UniProt sequence did not match that in PDB at', seq_pos)
                print('Skip to the next protein ...')
                valid_case = False
                break

            if args.database == 'AlphaFold':
                plddt = res['CA'].get_bfactor()
            else:
                plddt = 100
            # if plddt < 50:
            #     # skip residues of very low confidence
            #     continue

            contact_res = indexed_contacts[res]
            num_contacts = len(contact_res)
            contacts_pdb_pos = [r.get_id()[1] for r in contact_res]
            seq_seps = ';'.join(
                str(x) for x in [i - seq_pos for i in contacts_pdb_pos]
            )

            mis_var_sites = site_variability_missense.setdefault(seq_pos, 0)
            total_mis_sites = cds_ns_sites[seq_pos - 1][0]
            syn_var_sites = site_variability_synonymous.setdefault(seq_pos, 0)
            total_syn_sites = cds_ns_sites[seq_pos - 1][1]
            total_missense_obs = missense_counts.setdefault(seq_pos, 0)
            total_synonymous_obs = synonymous_counts.setdefault(seq_pos, 0)
            total_missense_poss = all_cds_ns_counts[seq_pos - 1][0]
            total_synonyms_poss = all_cds_ns_counts[seq_pos - 1][1]
            total_synonymous_rate = codon_mutation_rates[seq_pos - 1][0]
            total_missense_rate = codon_mutation_rates[seq_pos - 1][1]
            for j in contacts_pdb_pos:
                # count the total # observed variants in contacting residues
                mis_var_sites += site_variability_missense.setdefault(j, 0)
                syn_var_sites += site_variability_synonymous.setdefault(j, 0)
                total_missense_obs += missense_counts.setdefault(j, 0)
                total_synonymous_obs += synonymous_counts.setdefault(j, 0)

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

            mis_pmt_mean, mis_pmt_sd, mis_p_value, mis_pmt_mean_log, mis_pmt_sd_log = seq_utils.get_permutation_stats_log(
                mis_pmt_matrix, contacts_pdb_pos + [seq_pos], total_missense_obs
            )
            syn_pmt_mean, syn_pmt_sd, syn_p_value, syn_pmt_mean_log, syn_pmt_sd_log = seq_utils.get_permutation_stats_log(
                syn_pmt_matrix, contacts_pdb_pos + [seq_pos], total_synonymous_obs
            )

            # compute the fraction of expected missense variants
            cosmis.append(
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
                    '{:.3f}'.format((total_missense_obs - mis_pmt_mean) / mis_pmt_sd),  # Added cosmis calculation
                    '{:.3e}'.format(mis_p_value),
                    '{:.3f}'.format(syn_pmt_mean),
                    '{:.3f}'.format(syn_pmt_sd),
                    '{:.3e}'.format(syn_p_value),
                    enst_mp_counts[right_enst][2],
                    enst_mp_counts[right_enst][4],
                    total_exp_syn_counts,
                    total_exp_mis_counts,
                    '{:.2f}'.format(plddt),
                    len(pep_seq)
                ]
            )

        if not valid_case:
            continue

        with open(
                file=os.path.join(output_dir, uniprot_id + '_cosmis.tsv'),
                mode='wt'
        ) as opf:
            csv_writer = csv.writer(opf, delimiter='\t')
            csv_writer.writerow(get_dataset_headers())
            csv_writer.writerows(cosmis)

        logging.info(f'Successfully computed COSMIS for {uniprot_id}!')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3


import os
import numpy as np
from Bio.PDB import NeighborSearch

from hsc.utils import Contact


def search_for_all_contacts(residues, radius=8.0):
    """
    Search for all contacts in the given set of residues based on
    distances between CB atoms.

    Parameters
    ----------
    residues : list
        A list of Biopython Residue objects.
    radius : float
        The radius within which two residues are considered in contact.

    Returns
    -------
    list
        A list of Contact objects.

    """
    atom_list = []
    for r in residues:
        if r.get_resname() == 'GLY':
            try:
                atom_list.append(r['CA'])
            except KeyError:
                print('No CA atom found for GLY:', r, 'skipped ...')
                continue
        else:
            try:
                atom_list.append(r['CB'])
            except KeyError:
                print('No CB atom found for:', r.get_resname(), 'skipped ...')
                continue
            # atom_list += [a for a in r.get_atoms() if a.get_name()
            #               not in BACKBONE_ATOMS]
        # atom_list += [a for a in r.get_atoms()]
    ns = NeighborSearch(atom_list)
    all_contacts = [
        Contact(res_a=c[0], res_b=c[1])
        for c in ns.search_all(radius, level='R')
    ]
    return all_contacts

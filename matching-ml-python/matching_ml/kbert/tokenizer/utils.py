import json
import numpy as np

import pandas as pd

from kbert.tokenizer.constants import RANDOM_STATE


def molecules_from_texts(texts):
    molecules = pd.DataFrame([json.loads(text) for text in texts])
    return molecules


def group_by_index(statements):
    return statements.groupby(statements.index)


def add_statement_texts(statements: pd.DataFrame) -> pd.DataFrame:
    """
    For subject statements, computes texts by concatenating subject + ' ' + predicate, for object statements by
    concatenating predicate + ' ' + object
    :param statements: pandas Dataframe with following columns:
    - p: text of statement's predicate
    - n: text of statement's neighbor (object or subject)
    - r: 's' if n is a subject, 'o' if it is an object
    :return: statements dataframe with new column 'text' containing textual representation of the statement
    """
    has_statement_subject = statements['r'] == 's'
    statements.loc[has_statement_subject, 'text'] = \
        statements.loc[has_statement_subject, 'text_n'] + ' ' + statements.loc[has_statement_subject, 'text_p']
    statements.loc[has_statement_subject, 'tokens'] = \
        statements.loc[has_statement_subject, 'tokens_n'] + statements.loc[has_statement_subject, 'tokens_p']

    has_statement_object = statements['r'] == 'o'
    statements.loc[has_statement_object, 'text'] = \
        statements.loc[has_statement_object, 'text_p'] + ' ' + statements.loc[has_statement_object, 'text_n']
    statements.loc[has_statement_object, 'tokens'] = \
        statements.loc[has_statement_object, 'tokens_p'] + statements.loc[has_statement_object, 'tokens_n']
    return statements


def mask_array_with_minus_1(array, mask):
    return np.where(mask, array, -1)


def get_input_ids_by_statement_by_role_by_molecule(molecules, roles,
                                                   shape_token_by_statement_by_role_by_molecule, statements):
    # get tensor with input ids of statements by role by molecule
    input_id_by_statement_by_role_by_molecule = \
        np.zeros(shape_token_by_statement_by_role_by_molecule, dtype=int) - 1
    # fill in input ids of statement tokens
    statement_groups = group_by_index(statements)
    molecules['n_statement_tokens'] = statement_groups['n_tokens'].sum()
    z_molecule_4_roles_4_statements_4_tokens = np.concatenate(
        [np.repeat(i, n_statement_tokens) for i, n_statement_tokens in molecules['n_statement_tokens'].iteritems()])
    y_role_4_statements_4_tokens = np.concatenate(
        [np.repeat(i, n_statement_tokens) for i, n_statement_tokens in roles['n_statement_tokens'].iteritems()])
    statements['position_within_role_within_molecule'] = np.concatenate(
        [np.arange(i) for i in roles['n_statements'].values])
    x_statement_4_tokens = np.concatenate([
        np.repeat(position_within_role_within_molecule, n_tokens)
        for n_tokens, position_within_role_within_molecule in
        statements[['n_tokens', 'position_within_role_within_molecule']].values
    ])
    w_tokens = np.concatenate([np.arange(n_tokens) for n_tokens in statements['n_tokens'].values])
    input_id_by_statement_by_role_by_molecule[
        z_molecule_4_roles_4_statements_4_tokens,
        y_role_4_statements_4_tokens,
        x_statement_4_tokens,
        w_tokens
    ] = np.concatenate(statements['tokens'].values)
    return input_id_by_statement_by_role_by_molecule


def get_fits_token_in_input_by_statement_by_role_by_molecule(
        n_molecules, input_id_by_statement_by_role_by_molecule, shape_token_by_statement_by_molecule,
        shape_token_by_statement_by_role_by_molecule, max_statements_by_molecule,
        max_statements_by_role_by_molecule, max_tokens_by_molecule, max_tokens_by_statement,
        statement_offset_by_molecule, max_length):
    # 1. flip subject statements so that when a subject statement does not fit, leading tokens are cropped instead
    # of trailing ones
    mask = (input_id_by_statement_by_role_by_molecule != -1)
    mask[:, 1, :, :] = np.flip(mask[:, 1, :, :], axis=2)
    # 2. Flatten roles to sample across both roles
    mask = mask.reshape(shape_token_by_statement_by_molecule)
    # 3. shuffle
    statement_orders_by_molecule = \
        np.tile(np.arange(max_statements_by_molecule)[np.newaxis, :], (n_molecules, 1))
    for r in statement_orders_by_molecule:
        RANDOM_STATE.shuffle(r)
    z_molecule_4_statements_4_tokens = np.repeat(np.arange(n_molecules), max_tokens_by_molecule)
    y_statement_4_tokens = np.repeat(statement_orders_by_molecule, max_tokens_by_statement)
    x_tokens = np.tile(np.arange(max_tokens_by_statement),
                       n_molecules * 2 * max_statements_by_role_by_molecule)
    mask = mask[
        z_molecule_4_statements_4_tokens,
        y_statement_4_tokens,
        x_tokens
    ].reshape(shape_token_by_statement_by_molecule)
    # 4. Compute flags determining which tokens fit in input and which not
    # count through all statement tokens within each molecule to figure out which tokens still fit in the input
    offset_token_by_statement_by_molecule = np.where(
        mask,
        mask.reshape((n_molecules, max_tokens_by_molecule)).cumsum(
            axis=1)
        .reshape(shape_token_by_statement_by_molecule) +
        statement_offset_by_molecule[:, np.newaxis, np.newaxis],
        -1
    )
    fits_token_in_input_by_statement_by_molecule = \
        (offset_token_by_statement_by_molecule < max_length) & mask
    # 5. Shuffle back
    reverse_statement_orders_by_molecule = np.zeros(statement_orders_by_molecule.shape, dtype=int)
    y_molecule_4_statements = np.repeat(np.arange(n_molecules), max_statements_by_molecule)
    x_statements = statement_orders_by_molecule.ravel()
    reverse_statement_orders_by_molecule[y_molecule_4_statements, x_statements] = \
        np.tile(np.arange(max_statements_by_molecule), n_molecules)
    y_statement_4_tokens = np.repeat(reverse_statement_orders_by_molecule, max_tokens_by_statement)
    fits_token_in_input_by_statement_by_molecule = fits_token_in_input_by_statement_by_molecule[
        z_molecule_4_statements_4_tokens,
        y_statement_4_tokens,
        x_tokens
    ].reshape(shape_token_by_statement_by_molecule)
    # 6. unflatten roles
    fits_token_in_input_by_statement_by_role_by_molecule = \
        fits_token_in_input_by_statement_by_molecule.reshape(shape_token_by_statement_by_role_by_molecule)
    # 7. flip subject statement tokens back
    fits_token_in_input_by_statement_by_role_by_molecule[:, 1, :, :] = \
        np.flip(fits_token_in_input_by_statement_by_role_by_molecule[:, 1, :, :], axis=2)
    return fits_token_in_input_by_statement_by_role_by_molecule


def get_target_and_statement_token_ids(molecules, statements, targets, max_length):
    # initialize target input id tensor
    n_molecules = molecules.shape[0]
    target_groups = group_by_index(targets)
    molecules['n_targets'] = target_groups.size()
    max_targets_by_molecule = molecules['n_targets'].max()
    max_tokens_by_target = targets['n_tokens'].max()
    shape_token_by_target_by_molecule = (n_molecules, max_targets_by_molecule, max_tokens_by_target)
    input_ids_by_target_by_molecule = np.zeros(shape_token_by_target_by_molecule, dtype=int) - 1
    # fill in target input ids
    molecules['n_target_tokens'] = target_groups['n_tokens'].sum()
    z_molecule_4_targets_4_tokens = np.concatenate(
        [np.repeat(i, n_target_tokens) for i, n_target_tokens in molecules['n_target_tokens'].iteritems()])
    targets['position_within_molecule'] = np.concatenate([np.arange(i) for i in molecules['n_targets'].values])
    y_target_4_tokens = np.concatenate([
        np.repeat(position_within_molecule, n_tokens)
        for n_tokens, position_within_molecule in targets[['n_tokens', 'position_within_molecule']].values
    ])
    x_tokens = np.concatenate([np.arange(n_tokens) for n_tokens in targets['n_tokens'].values])
    input_ids_by_target_by_molecule[
        z_molecule_4_targets_4_tokens,
        y_target_4_tokens,
        x_tokens
    ] = np.concatenate(targets['tokens'].values)
    # compute mask for tokens that fit in input
    target_token_mask = (input_ids_by_target_by_molecule != -1)
    offset_token_by_target_by_molecule = np.where(
        target_token_mask,
        target_token_mask.reshape(
            (n_molecules, max_targets_by_molecule * max_tokens_by_target)
        ).cumsum(axis=1).reshape(shape_token_by_target_by_molecule), -1)
    fits_token_in_input_by_target_by_molecule = \
        (offset_token_by_target_by_molecule < max_length) & target_token_mask
    offset_token_by_target_by_molecule = mask_array_with_minus_1(
        array=offset_token_by_target_by_molecule,
        mask=fits_token_in_input_by_target_by_molecule
    )
    input_ids_by_target_by_molecule = mask_array_with_minus_1(
        array=input_ids_by_target_by_molecule,
        mask=fits_token_in_input_by_target_by_molecule
    )
    # get roles (groups of subject/object statements within molecules)
    role_groups = statements.groupby([statements.index, statements['r']])
    roles = role_groups.size().rename('n_statements').to_frame()
    roles['n_statement_tokens'] = role_groups['n_tokens'].sum()
    roles = roles.reset_index(level=0, drop=True)
    roles.index = roles.index.map({'o': 0, 's': 1})
    max_statements_by_role_by_molecule = roles['n_statements'].max()
    max_tokens_by_statement = statements['n_tokens'].max()
    shape_token_by_statement_by_role_by_molecule = (
        n_molecules, 2, max_statements_by_role_by_molecule, max_tokens_by_statement)
    input_ids_by_statement_by_role_by_molecule = get_input_ids_by_statement_by_role_by_molecule(
        molecules, roles, shape_token_by_statement_by_role_by_molecule, statements)
    max_statements_by_molecule = 2 * max_statements_by_role_by_molecule
    max_tokens_by_molecule = max_statements_by_molecule * max_tokens_by_statement
    shape_token_by_statement_by_molecule = (n_molecules, max_statements_by_molecule, max_tokens_by_statement)
    statement_offset_by_molecule = offset_token_by_target_by_molecule.max(axis=(1, 2))
    # Sample statements
    fits_token_in_input_by_statement_by_role_by_molecule = \
        get_fits_token_in_input_by_statement_by_role_by_molecule(
            n_molecules, input_ids_by_statement_by_role_by_molecule, shape_token_by_statement_by_molecule,
            shape_token_by_statement_by_role_by_molecule, max_statements_by_molecule,
            max_statements_by_role_by_molecule, max_tokens_by_molecule, max_tokens_by_statement,
            statement_offset_by_molecule, max_length)
    fits_token_in_input_by_statement_by_molecule = \
        fits_token_in_input_by_statement_by_role_by_molecule.reshape(shape_token_by_statement_by_molecule)
    fits_statement_in_input_by_molecule = fits_token_in_input_by_statement_by_molecule.any(2)
    # mask statement tokens that do not fit in input
    input_ids_by_statement_by_role_by_molecule = mask_array_with_minus_1(
        array=input_ids_by_statement_by_role_by_molecule,
        mask=fits_token_in_input_by_statement_by_role_by_molecule
    )
    return \
        n_molecules, \
        input_ids_by_target_by_molecule, \
        input_ids_by_statement_by_role_by_molecule, \
        fits_statement_in_input_by_molecule, \
        fits_token_in_input_by_statement_by_molecule, \
        fits_token_in_input_by_statement_by_role_by_molecule, \
        fits_token_in_input_by_target_by_molecule, \
        max_targets_by_molecule, \
        max_tokens_by_molecule, \
        offset_token_by_target_by_molecule, \
        shape_token_by_statement_by_molecule, \
        shape_token_by_statement_by_role_by_molecule, \
        statement_offset_by_molecule

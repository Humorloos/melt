import json
import numpy as np

import pandas as pd


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

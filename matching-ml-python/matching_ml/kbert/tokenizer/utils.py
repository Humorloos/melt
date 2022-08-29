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
        statement_offset_by_molecule, max_length_by_molecule):
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
        (offset_token_by_statement_by_molecule <= max_length_by_molecule[:, np.newaxis, np.newaxis]) & mask
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


def get_target_and_statement_token_ids(
        molecules,
        statements,
        targets,
        max_length_by_molecule,
        n_molecules,
        target_offset_by_molecule,
        max_targets_by_molecule=None,
):
    # initialize target input id tensor
    if max_targets_by_molecule is None:
        max_targets_by_molecule = molecules['n_targets'].max()
    max_tokens_by_target = targets['n_tokens'].max()
    shape_token_by_target_by_molecule = (n_molecules, max_targets_by_molecule, max_tokens_by_target)
    input_ids_by_target_by_molecule = np.zeros(shape_token_by_target_by_molecule, dtype=int) - 1
    # fill in target input ids
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
        target_token_mask
        .reshape((n_molecules, max_targets_by_molecule * max_tokens_by_target))
        .cumsum(axis=1).reshape(shape_token_by_target_by_molecule),
        -1
    )
    fits_token_in_input_by_target_by_molecule = \
        (offset_token_by_target_by_molecule <= max_length_by_molecule[:, np.newaxis, np.newaxis]) & target_token_mask
    offset_token_by_target_by_molecule = mask_array_with_minus_1(
        array=offset_token_by_target_by_molecule + target_offset_by_molecule[:, np.newaxis, np.newaxis] - 1,
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
    statement_offset_by_molecule = offset_token_by_target_by_molecule.max(axis=(1, 2)) - target_offset_by_molecule + 1
    # Sample statements
    fits_token_in_input_by_statement_by_role_by_molecule = \
        get_fits_token_in_input_by_statement_by_role_by_molecule(
            n_molecules, input_ids_by_statement_by_role_by_molecule, shape_token_by_statement_by_molecule,
            shape_token_by_statement_by_role_by_molecule, max_statements_by_molecule,
            max_statements_by_role_by_molecule, max_tokens_by_molecule, max_tokens_by_statement,
            statement_offset_by_molecule, max_length_by_molecule)
    fits_token_in_input_by_statement_by_molecule = \
        fits_token_in_input_by_statement_by_role_by_molecule.reshape(shape_token_by_statement_by_molecule)
    is_statement_in_input_by_molecule = fits_token_in_input_by_statement_by_molecule.any(2)
    # mask statement tokens that do not fit in input
    input_ids_by_statement_by_role_by_molecule = mask_array_with_minus_1(
        array=input_ids_by_statement_by_role_by_molecule,
        mask=fits_token_in_input_by_statement_by_role_by_molecule
    )
    return \
        input_ids_by_target_by_molecule, \
        input_ids_by_statement_by_role_by_molecule, \
        is_statement_in_input_by_molecule, \
        fits_token_in_input_by_statement_by_molecule, \
        fits_token_in_input_by_statement_by_role_by_molecule, \
        fits_token_in_input_by_target_by_molecule, \
        max_tokens_by_molecule, \
        offset_token_by_target_by_molecule, \
        shape_token_by_statement_by_molecule, \
        shape_token_by_statement_by_role_by_molecule


def count_tokens(molecules, statements, targets):
    target_groups = group_by_index(targets)
    molecules['n_targets'] = target_groups.size()
    molecules['n_target_tokens'] = target_groups['n_tokens'].sum()
    statement_groups = group_by_index(statements)
    molecules['n_statement_tokens'] = statement_groups['n_tokens'].sum()
    molecules['n_tokens'] = molecules['n_tokens'] = molecules['n_statement_tokens'] + molecules['n_target_tokens']


def fill_in_molecule_input_ids(
        fits_token_in_input_by_statement_by_role_by_molecule,
        fits_token_in_input_by_target_by_molecule,
        input_ids,
        input_ids_by_statement_by_role_by_molecule,
        input_ids_by_target_by_molecule,
        max_tokens_by_molecule,
        n_molecules,
        offset_token_by_target_by_molecule,
        shape_token_by_statement_by_role_by_molecule,
        statement_offset_by_molecule
):
    # fill in target token input ids in input ids array
    n_target_tokens_by_molecule = fits_token_in_input_by_target_by_molecule.sum((1, 2))
    y_molecule_4_target_tokens = np.concatenate(
        [np.repeat(i, n) for i, n in enumerate(n_target_tokens_by_molecule)])
    x_target_tokens = np.concatenate(
        [offsets_tokens_by_target[offsets_tokens_by_target != -1] for offsets_tokens_by_target in
         offset_token_by_target_by_molecule])
    input_ids[y_molecule_4_target_tokens, x_target_tokens] = np.concatenate(
        [token_by_target[token_by_target != -1] for token_by_target in
         input_ids_by_target_by_molecule])
    # fill in input ids of statement tokens into input ids tensor
    n_statement_tokens_by_molecule = fits_token_in_input_by_statement_by_role_by_molecule.sum((1, 2, 3))
    y_molecule_4_statement_tokens = np.concatenate(
        [np.repeat(i, n) for i, n in enumerate(n_statement_tokens_by_molecule)])
    offset_token_by_statement_by_role_by_molecule = np.where(
        fits_token_in_input_by_statement_by_role_by_molecule,
        fits_token_in_input_by_statement_by_role_by_molecule.reshape(
            (n_molecules, max_tokens_by_molecule)
        ).cumsum(axis=1).reshape(shape_token_by_statement_by_role_by_molecule) +
        statement_offset_by_molecule[:, np.newaxis, np.newaxis, np.newaxis] - 1,
        -1
    )
    x_statement_tokens = np.concatenate(
        [offsets_tokens_by_statement[offsets_tokens_by_statement != -1] for offsets_tokens_by_statement in
         offset_token_by_statement_by_role_by_molecule])
    input_ids[y_molecule_4_statement_tokens, x_statement_tokens] = np.concatenate([
        input_id_by_statement[input_id_by_statement != -1]
        for input_id_by_statement in input_ids_by_statement_by_role_by_molecule
    ])
    return n_statement_tokens_by_molecule, \
           n_target_tokens_by_molecule, \
           offset_token_by_statement_by_role_by_molecule, \
           x_target_tokens, \
           y_molecule_4_target_tokens


def fill_in_molecule_position_ids(
        fits_token_in_input_by_statement_by_role_by_molecule,
        fits_token_in_input_by_target_by_molecule,
        offset_token_by_statement_by_role_by_molecule,
        position_ids,
        x_target_tokens,
        y_molecule_4_target_tokens,
        position_offset_first_token_by_molecule
):
    # Compute position ids of subject statement tokens, subject statements that are shorter than the longest one do
    # not start at position one, but at a position such that the last token is adjacent to the first target token
    fits_token_in_input_by_subject_statement_by_molecule = \
        fits_token_in_input_by_statement_by_role_by_molecule[:, 1, :, :]
    n_tokens_by_subject_statement_by_molecule = \
        fits_token_in_input_by_subject_statement_by_molecule.sum(2)
    max_subject_statement_tokens_by_molecule = n_tokens_by_subject_statement_by_molecule.max(1)
    offset_first_token_by_subject_statement_by_molecule = \
        max_subject_statement_tokens_by_molecule[:, np.newaxis] \
        - n_tokens_by_subject_statement_by_molecule \
        + position_offset_first_token_by_molecule[:, np.newaxis] \
        - 1
    cumsum_token_by_subject_statement_by_molecule = \
        fits_token_in_input_by_subject_statement_by_molecule.cumsum(2)
    subject_statement_token_positions = np.where(
        fits_token_in_input_by_subject_statement_by_molecule,
        cumsum_token_by_subject_statement_by_molecule +
        offset_first_token_by_subject_statement_by_molecule[:, :, np.newaxis],
        -1
    )
    # fill in position ids of subject statements
    n_statement_tokens_by_role_by_molecule = fits_token_in_input_by_statement_by_role_by_molecule.sum((2, 3))
    y_molecule_4_subject_statement_tokens = np.concatenate(
        [np.repeat(i, n) for i, n in enumerate(n_statement_tokens_by_role_by_molecule[:, 1])])
    x_subject_statement_tokens = np.concatenate([
        offsets_tokens_by_statement[offsets_tokens_by_statement != -1]
        for offsets_tokens_by_statement in offset_token_by_statement_by_role_by_molecule[:, 1]])
    position_ids[y_molecule_4_subject_statement_tokens, x_subject_statement_tokens] = \
        subject_statement_token_positions[subject_statement_token_positions != -1]
    # fill in position ids of targets
    cumsum_token_by_target_by_molecule = fits_token_in_input_by_target_by_molecule.cumsum(2)
    target_token_position_offset_by_molecule = max_subject_statement_tokens_by_molecule \
                                               + position_offset_first_token_by_molecule \
                                               - 1
    target_token_positions = np.where(
        fits_token_in_input_by_target_by_molecule,
        # This cannot be done with max() as for object statements, because not every molecule has subject statements.
        # For molecules without subject statements, max() would be -1 then
        cumsum_token_by_target_by_molecule + target_token_position_offset_by_molecule[:, np.newaxis, np.newaxis],
        -1
    )
    position_ids[y_molecule_4_target_tokens, x_target_tokens] = target_token_positions[target_token_positions != -1]
    # compute position ids of object statements
    fits_token_in_input_by_object_statement_by_molecule = \
        fits_token_in_input_by_statement_by_role_by_molecule[:, 0, :, :]
    cumsum_token_by_object_statement_by_molecule = \
        fits_token_in_input_by_object_statement_by_molecule.cumsum(2)
    max_target_token_position_by_molecule = target_token_positions.max((1, 2))
    object_statement_token_positions = np.where(
        fits_token_in_input_by_object_statement_by_molecule,
        cumsum_token_by_object_statement_by_molecule +
        max_target_token_position_by_molecule[:, np.newaxis, np.newaxis],
        -1)
    # fill in object statement position ids
    y_molecule_for_object_statement_tokens = np.concatenate([
        np.repeat(i, n) for i, n in enumerate(n_statement_tokens_by_role_by_molecule[:, 0])
    ])
    x_object_statement_tokens = np.concatenate([
        offsets_tokens_by_statement[offsets_tokens_by_statement != -1]
        for offsets_tokens_by_statement in offset_token_by_statement_by_role_by_molecule[:, 0]
    ])
    position_ids[y_molecule_for_object_statement_tokens, x_object_statement_tokens] = \
        object_statement_token_positions[object_statement_token_positions != -1]


def add_molecule_seeing_token_holes(
        attention_masks,
        offset_token_by_statement_by_role_by_molecule,
        offset_token_by_target_by_molecule,
        target_offset_by_molecule,
        token_offset_by_molecule
):
    seq_length_by_molecule = np.concatenate([
        a[:, np.newaxis] for a in [
            offset_token_by_target_by_molecule.max((1, 2)),
            offset_token_by_statement_by_role_by_molecule.max((1, 2, 3))
        ]
    ], axis=1).max(1) + 1 - target_offset_by_molecule
    z_molecule_4_tokens_4_tokens = np.concatenate([
        np.repeat(i, seq_length)
        for i, seq_length in enumerate(seq_length_by_molecule)
    ])
    y_tokens_4_tokens = np.concatenate([
        np.arange(target_offset, seq_length + target_offset)
        for seq_length, target_offset in zip(seq_length_by_molecule, target_offset_by_molecule)
    ])
    x_tokens = np.concatenate([np.repeat(token_offset, seq_length)
                               for token_offset, seq_length in zip(token_offset_by_molecule, seq_length_by_molecule)])
    attention_masks[z_molecule_4_tokens_4_tokens, y_tokens_4_tokens, x_tokens] = 1
    attention_masks[z_molecule_4_tokens_4_tokens, x_tokens, y_tokens_4_tokens] = 1
    return seq_length_by_molecule


def add_molecule_holes(attention_masks, is_statement_in_input_by_molecule,
                       fits_token_in_input_by_statement_by_molecule, fits_token_in_input_by_target_by_molecule,
                       max_length_by_molecule, n_statement_tokens_by_molecule, n_target_tokens_by_molecule,
                       offset_token_by_statement_by_role_by_molecule, offset_token_by_target_by_molecule,
                       seq_length_by_molecule, shape_token_by_statement_by_molecule, target_offset_by_molecule,
                       statement_offset_by_molecule):
    # each target sees all statements and is seen by all statements
    n_target_tokens_times_n_statement_tokens_by_molecule = \
        n_target_tokens_by_molecule * n_statement_tokens_by_molecule
    z_molecule_4_tokens_4_tokens = np.concatenate([
        np.repeat(i, n_target_tokens_times_n_statement_tokens)
        for i, n_target_tokens_times_n_statement_tokens
        in enumerate(n_target_tokens_times_n_statement_tokens_by_molecule)
    ])
    y_tokens_4_tokens = np.concatenate([
        np.repeat(np.arange(statement_offset, n_statement_tokens + statement_offset), n_target_tokens)
        for n_target_tokens, n_statement_tokens, statement_offset
        in zip(n_target_tokens_by_molecule, n_statement_tokens_by_molecule, statement_offset_by_molecule)
    ])
    x_tokens = np.concatenate([
        np.tile(np.arange(target_offset, target_offset + n_target_tokens), n_statement_tokens)
        for n_target_tokens, target_offset, n_statement_tokens
        in zip(n_target_tokens_by_molecule, target_offset_by_molecule, n_statement_tokens_by_molecule)
    ])
    attention_masks[z_molecule_4_tokens_4_tokens, y_tokens_4_tokens, x_tokens] = 1
    attention_masks[z_molecule_4_tokens_4_tokens, x_tokens, y_tokens_4_tokens] = 1
    # add holes for targets (target tokens can see tokens of same target)
    n_tokens_by_target_by_molecule = fits_token_in_input_by_target_by_molecule.sum(2)
    sum_square_target_tokens_by_molecule = (n_tokens_by_target_by_molecule ** 2).sum(1)
    z_molecule_4_tokens_4_tokens = np.concatenate([
        np.repeat(i, sum_square_target_tokens)
        for i, sum_square_target_tokens in enumerate(sum_square_target_tokens_by_molecule)
    ])
    is_target_in_input_by_molecule = fits_token_in_input_by_target_by_molecule.any(2)
    mask_offset_by_every_target_by_molecule = np.where(
        fits_token_in_input_by_target_by_molecule,
        offset_token_by_target_by_molecule,
        (max_length_by_molecule + target_offset_by_molecule - 1)[:, np.newaxis, np.newaxis]
    ).min(2)
    mask_offset_by_participating_target_by_molecule = np.where(
        is_target_in_input_by_molecule,
        mask_offset_by_every_target_by_molecule,
        -1
    )
    mask_offset_next_by_target_by_molecule = np.where(
        is_target_in_input_by_molecule,
        offset_token_by_target_by_molecule.max(2) + 1,
        -1
    )
    y_tokens_4_tokens = np.concatenate([
        np.repeat(np.arange(offset, offset_next), offset_next - offset)
        for offset_by_target, offset_next_by_target
        in zip(mask_offset_by_participating_target_by_molecule, mask_offset_next_by_target_by_molecule)
        for offset, offset_next
        in zip(offset_by_target, offset_next_by_target)
    ])
    x_tokens = np.concatenate([
        np.tile(np.arange(offset, offset_next), offset_next - offset)
        for offset_by_target, offset_next_by_target
        in zip(mask_offset_by_participating_target_by_molecule, mask_offset_next_by_target_by_molecule)
        for offset, offset_next
        in zip(offset_by_target, offset_next_by_target)
    ])
    attention_masks[z_molecule_4_tokens_4_tokens, y_tokens_4_tokens, x_tokens] = 1
    # add holes for statements (like in targets, tokens can see each other)
    n_tokens_by_statement_by_molecule = fits_token_in_input_by_statement_by_molecule.sum(2)
    sum_square_statement_tokens_by_molecule = (n_tokens_by_statement_by_molecule ** 2).sum((1))
    z_molecule_4_tokens_4_tokens = np.concatenate([
        np.repeat(i, sum_square_statement_tokens)
        for i, sum_square_statement_tokens in enumerate(sum_square_statement_tokens_by_molecule)
    ])
    offset_token_by_statement_by_molecule = \
        offset_token_by_statement_by_role_by_molecule.reshape(shape_token_by_statement_by_molecule)
    mask_offset_by_every_statement_by_molecule = np.where(
        fits_token_in_input_by_statement_by_molecule,
        offset_token_by_statement_by_molecule,
        (max_length_by_molecule + target_offset_by_molecule - 1)[:, np.newaxis, np.newaxis]
    ).min(2)
    mask_offset_by_participating_statement_by_molecule = np.where(
        is_statement_in_input_by_molecule,
        mask_offset_by_every_statement_by_molecule,
        -1
    )
    mask_offset_next_by_statement_by_molecule = np.where(
        is_statement_in_input_by_molecule,
        offset_token_by_statement_by_molecule.max(2) + 1,
        -1
    )
    y_tokens_4_tokens = np.concatenate([
        np.repeat(np.arange(offset, offset_next), offset_next - offset)
        for offset_by_statement, offset_next_by_statement
        in zip(mask_offset_by_participating_statement_by_molecule, mask_offset_next_by_statement_by_molecule)
        for offset, offset_next
        in zip(offset_by_statement, offset_next_by_statement)
    ])
    x_tokens = np.concatenate([
        np.tile(np.arange(offset, offset_next), offset_next - offset)
        for offset_by_statement, offset_next_by_statement
        in zip(mask_offset_by_participating_statement_by_molecule, mask_offset_next_by_statement_by_molecule)
        for offset, offset_next
        in zip(offset_by_statement, offset_next_by_statement)
    ])
    attention_masks[z_molecule_4_tokens_4_tokens, y_tokens_4_tokens, x_tokens] = 1
    return n_tokens_by_target_by_molecule

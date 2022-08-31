from inspect import getmembers
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import TensorType
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TruncationStrategy, TextInputPair, \
    PreTokenizedInputPair, EncodedInput, EncodedInputPair
from transformers.utils import PaddingStrategy

import kbert.tokenizer.utils
import kbert.tokenizer.utils
from kbert.tokenizer.utils import add_statement_texts
from kbert.tokenizer.utils import molecules_from_texts


class TMTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, index_files: List[Path] = None, *inputs, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        return TMTokenizer(tokenizer, index_files)

    def __init__(self, tokenizer, index_files: List[Path] = None):
        self.base_tokenizer = tokenizer

        self.token_index = pd.DataFrame(columns=['text', 'tokens', 'n_tokens'])
        if index_files is not None:
            for f in index_files:
                self.extend_index(f)

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs
    ):
        if max_length is None:
            max_length = self.base_tokenizer.max_len_single_sentence
        # get molecules
        molecules = molecules_from_texts(text)
        # get targets
        targets = self.targets_from_molecules(molecules)

        # get statements
        statements = self.statements_from_molecules(molecules)

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
        statement_offset_by_molecule = kbert.tokenizer.utils.get_target_and_statement_token_ids(molecules, statements,
                                                                                                targets, max_length)

        # initialize input id tensor
        shape_molecules_by_max_seq_length = (n_molecules, max_length)
        input_ids = np.zeros(shape_molecules_by_max_seq_length, dtype=int)

        # fill in cls token ids
        cls_tokens_y = np.arange(n_molecules)
        cls_tokens_x = np.zeros(n_molecules, dtype=int)
        input_ids[cls_tokens_y, cls_tokens_x] = self.base_tokenizer.cls_token_id

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
            statement_offset_by_molecule[:, np.newaxis, np.newaxis, np.newaxis],
            -1
        )
        x_statement_tokens = np.concatenate(
            [offsets_tokens_by_statement[offsets_tokens_by_statement != -1] for offsets_tokens_by_statement in
             offset_token_by_statement_by_role_by_molecule])
        input_ids[y_molecule_4_statement_tokens, x_statement_tokens] = np.concatenate([
            input_id_by_statement[input_id_by_statement != -1]
            for input_id_by_statement in input_ids_by_statement_by_role_by_molecule
        ])

        # Compute position ids
        position_ids = np.zeros(shape_molecules_by_max_seq_length, dtype=int)

        # Compute position ids of subject statement tokens, subject statements that are shorter than the longest one do
        # not start at position one, but at a position such that the last token is adjacent to the first target token
        fits_token_in_input_by_subject_statement_by_molecule = \
            fits_token_in_input_by_statement_by_role_by_molecule[:, 1, :, :]
        n_tokens_by_subject_statement_by_molecule = \
            fits_token_in_input_by_subject_statement_by_molecule.sum(2)
        max_subject_statement_tokens_by_molecule = n_tokens_by_subject_statement_by_molecule.max(1)
        offset_first_token_by_subject_statement_by_molecule = \
            max_subject_statement_tokens_by_molecule[:, np.newaxis] - n_tokens_by_subject_statement_by_molecule
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
        target_token_positions = np.where(
            fits_token_in_input_by_target_by_molecule,
            cumsum_token_by_target_by_molecule + max_subject_statement_tokens_by_molecule[:, np.newaxis, np.newaxis],
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

        # initialize attention masks
        attention_masks = np.zeros((n_molecules, *2 * [max_length]), dtype=int)

        # Add holes for cls token
        seq_length_by_molecule = np.concatenate([
            a[:, np.newaxis] for a in [
                offset_token_by_target_by_molecule.max((1, 2)),
                offset_token_by_statement_by_role_by_molecule.max((1, 2, 3))
            ]
        ], axis=1).max(1) + 1
        z_molecule_4_tokens_4_tokens = np.concatenate([
            np.repeat(i, seq_length)
            for i, seq_length in enumerate(seq_length_by_molecule)
        ])
        y_tokens_4_tokens = np.concatenate([
            np.arange(seq_length) for seq_length in seq_length_by_molecule
        ])
        x_tokens = np.zeros(seq_length_by_molecule.sum(), dtype=int)
        attention_masks[z_molecule_4_tokens_4_tokens, y_tokens_4_tokens, x_tokens] = 1
        attention_masks[z_molecule_4_tokens_4_tokens, x_tokens, y_tokens_4_tokens] = 1

        # each target sees all statements and is seen by all statements
        n_statement_seeing_tokens_by_molecule = n_target_tokens_by_molecule + 1
        n_target_tokens_times_n_statement_tokens_by_molecule = \
            n_target_tokens_by_molecule * n_statement_tokens_by_molecule
        z_molecule_4_tokens_4_tokens = np.concatenate([
            np.repeat(i, n_target_tokens_times_n_statement_tokens)
            for i, n_target_tokens_times_n_statement_tokens
            in enumerate(n_target_tokens_times_n_statement_tokens_by_molecule)
        ])
        y_tokens_4_tokens = np.concatenate([
            np.repeat(np.arange(1, n_statement_seeing_tokens), n_statement_tokens)
            for n_statement_seeing_tokens, n_statement_tokens
            in zip(n_statement_seeing_tokens_by_molecule, n_statement_tokens_by_molecule)
        ])
        x_tokens = np.concatenate([
            np.tile(np.arange(n_statement_seeing_tokens, seq_length), n_target_tokens)
            for n_statement_seeing_tokens, seq_length, n_target_tokens
            in zip(n_statement_seeing_tokens_by_molecule, seq_length_by_molecule, n_target_tokens_by_molecule)
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
        fits_target_in_input_by_molecule = fits_token_in_input_by_target_by_molecule.any(2)
        offset_by_target_by_molecule = np.where(
            fits_target_in_input_by_molecule,
            np.where(
                fits_token_in_input_by_target_by_molecule,
                offset_token_by_target_by_molecule,
                max_length
            ).min(2),
            -1
        )
        offset_next_by_target_by_molecule = np.where(
            fits_target_in_input_by_molecule,
            offset_token_by_target_by_molecule.max(2) + 1,
            -1
        )
        y_tokens_4_tokens = np.concatenate([
            np.repeat(np.arange(offset, offset_next), offset_next - offset)
            for offset_by_target, offset_next_by_target
            in zip(offset_by_target_by_molecule, offset_next_by_target_by_molecule)
            for offset, offset_next
            in zip(offset_by_target, offset_next_by_target)
        ])
        x_tokens = np.concatenate([
            np.tile(np.arange(offset, offset_next), offset_next - offset)
            for offset_by_target, offset_next_by_target
            in zip(offset_by_target_by_molecule, offset_next_by_target_by_molecule)
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
        offset_by_statement_by_molecule = np.where(
            fits_statement_in_input_by_molecule,
            np.where(
                fits_token_in_input_by_statement_by_molecule,
                offset_token_by_statement_by_molecule,
                max_length
            ).min(2),
            -1
        )
        offset_next_by_statement_by_molecule = np.where(
            fits_statement_in_input_by_molecule,
            offset_token_by_statement_by_molecule.max(2) + 1,
            -1
        )
        y_tokens_4_tokens = np.concatenate([
            np.repeat(np.arange(offset, offset_next), offset_next - offset)
            for offset_by_statement, offset_next_by_statement
            in zip(offset_by_statement_by_molecule, offset_next_by_statement_by_molecule)
            for offset, offset_next
            in zip(offset_by_statement, offset_next_by_statement)
        ])
        x_tokens = np.concatenate([
            np.tile(np.arange(offset, offset_next), offset_next - offset)
            for offset_by_statement, offset_next_by_statement
            in zip(offset_by_statement_by_molecule, offset_next_by_statement_by_molecule)
            for offset, offset_next
            in zip(offset_by_statement, offset_next_by_statement)
        ])
        attention_masks[z_molecule_4_tokens_4_tokens, y_tokens_4_tokens, x_tokens] = 1

        # compute targets mask
        targets_mask = np.zeros((n_molecules, max_targets_by_molecule, max_length))
        z_molecule_4_targets_4_tokens = np.concatenate([
            np.repeat(i, n_target_tokens) for i, n_target_tokens in enumerate(n_target_tokens_by_molecule)
        ])
        y_target_4_tokens = np.concatenate([
            np.repeat(i, n_tokens)
            for n_tokens_by_target in n_tokens_by_target_by_molecule
            for i, n_tokens in enumerate(n_tokens_by_target)
        ])
        x_tokens = np.concatenate(
            [np.arange(1, n_target_tokens + 1) for n_target_tokens in n_target_tokens_by_molecule])
        targets_mask[z_molecule_4_targets_4_tokens, y_target_4_tokens, x_tokens] = 1
        return {
            'input_ids': torch.IntTensor(input_ids),
            'position_ids': torch.IntTensor(position_ids),
            'attention_mask': torch.IntTensor(attention_masks),
            'token_type_ids': torch.IntTensor(np.zeros(shape_molecules_by_max_seq_length)),
            'targets_mask': torch.IntTensor(targets_mask)
        }

    def extend_index(self, index_file):
        index_extension = pd.read_csv(index_file, index_col=0, names=['text'])
        index_extension['tokens'] = self.base_tokenizer.batch_encode_plus(
            index_extension['text'].tolist(), add_special_tokens=False
        ).input_ids
        index_extension['n_tokens'] = index_extension['tokens'].apply(len)
        new_index = pd.concat([self.token_index, index_extension])
        self.token_index = new_index[~new_index.index.duplicated()]

    def targets_from_molecules(self, molecules):
        targets = molecules.explode('t')[['t']]
        targets['t'] = targets['t'].astype('int64')
        targets = targets.merge(self.token_index[['text', 'tokens', 'n_tokens']], left_on='t', right_index=True).drop(
            columns='t')
        targets['statement_seeing'] = True
        targets['all_seeing'] = False
        targets['r'] = 't'
        return targets.sort_index()

    def statements_from_molecules(self, molecules):
        statement_dicts = molecules['s'].explode().dropna()
        statements = pd.DataFrame(statement_dicts.tolist(), index=statement_dicts.index)
        statements[['n', 'p']] = statements[['n', 'p']].astype('int64')
        statements = statements.merge(
            self.token_index,
            left_on='n',
            right_index=True,
        ).merge(
            self.token_index,
            left_on='p',
            right_index=True,
            suffixes=('_n', '_p')
        )
        statements = add_statement_texts(statements)
        statements['n_tokens'] = statements['n_tokens_n'] + statements['n_tokens_p']
        statements['statement_seeing'] = False
        statements['all_seeing'] = False
        return statements.rename_axis('molecule_id').sort_values(by=['molecule_id', 'r'])

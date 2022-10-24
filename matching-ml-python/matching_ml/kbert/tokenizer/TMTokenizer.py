import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer
from transformers import TensorType
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TruncationStrategy, BatchEncoding, \
    EncodedInput
from transformers.utils import PaddingStrategy
from typing import Union, List, Optional, Dict

from kbert.tokenizer.utils import add_statement_texts, get_target_and_statement_token_ids, count_tokens, \
    fill_in_molecule_input_ids, molecules_from_texts, fill_in_molecule_position_ids, add_molecule_seeing_token_holes, \
    add_molecule_holes


class TMTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, index_files: List[Path] = None, max_length=None,
                        tm_attention=True, *inputs, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        return TMTokenizer(tokenizer, index_files, max_length, tm_attention)

    def __init__(self, tokenizer, index_files: List[Path] = None, max_length=None, tm_attention=True):
        if max_length is not None:
            tokenizer.model_max_length = max_length

        self.base_tokenizer = tokenizer
        self.token_index = pd.DataFrame(columns=['text', 'tokens', 'n_tokens'])
        self.tm_attention = tm_attention
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
            max_length = self.base_tokenizer.model_max_length
        input_ids = []
        position_ids = []
        if self.tm_attention:
            attention_masks = []
        targets_mask = []
        token_type_ids = []
        fits_token_in_input_by_statement_by_molecule_pair, \
        is_statement_in_input_by_molecule_pair, \
        n_statement_tokens_by_molecule_pair, \
        n_target_tokens_by_molecule_pair, \
        shape_token_by_statement_by_molecule_pair, \
        fits_token_in_input_by_target_by_molecule_pair, \
        max_length_by_molecule_pair, \
        molecule_range, \
        offset_token_by_statement_by_role_by_molecule_pair, \
        offset_token_by_target_by_molecule_pair, \
        sep_token_offset_by_molecule, \
        statement_offset_by_molecule_pair, \
        target_offset_by_molecule_pair = 13 * [None]
        for batch_idx in range(((len(text) - 1) // 64) + 1):
            batch_text = text[batch_idx * 64:(batch_idx + 1) * 64]
            # get molecules
            molecules = molecules_from_texts(batch_text)
            # get targets
            targets = self.targets_from_molecules(molecules)

            # get statements
            statements = self.statements_from_molecules(molecules)
            count_tokens(molecules, statements, targets)

            n_molecules = molecules.shape[0]

            # do the same for pair molecules if there are any
            if text_pair is not None:
                batch_text_pair = text_pair[batch_idx * 64:(batch_idx + 1) * 64]
                molecules_pair = molecules_from_texts(batch_text_pair)
                targets_pair = self.targets_from_molecules(molecules_pair)
                statements_pair = self.statements_from_molecules(molecules_pair)
                count_tokens(molecules_pair, statements_pair, targets_pair)
                # -2 because we have 1 CLS and 1 SEP token
                max_example_length = max_length - 2
                max_molecule_length = max_example_length // 2
                greater_max_length_by_molecule_pair = molecules_pair['n_tokens'].values > max_molecule_length
                # max input ids for first input depend on whether second input is longer than half of the remaining input:
                # - if it is not longer, max input ids are what is left after filling in whole second input
                # - if it is longer, max input ids further depend on length of first input:
                #   - if it is longer than half of whole input, max input ids are half of the whole input
                #   - otherwise, they are same as input 1 length
                max_length_by_molecule = np.where(
                    greater_max_length_by_molecule_pair,
                    np.minimum(max_molecule_length, molecules['n_tokens'].values.astype(int)),
                    max_example_length - molecules_pair['n_tokens'].values.astype(int)
                )
                max_length_by_molecule_pair = max_example_length - max_length_by_molecule
            else:
                # -1 because of CLS token
                max_length_by_molecule = np.repeat(max_length - 1, n_molecules)

            max_targets_by_molecule = molecules['n_targets'].max()

            target_offset_by_molecule = np.repeat(1, n_molecules)
            input_ids_by_target_by_molecule, \
            input_ids_by_statement_by_role_by_molecule, \
            is_statement_in_input_by_molecule, \
            fits_token_in_input_by_statement_by_molecule, \
            fits_token_in_input_by_statement_by_role_by_molecule, \
            fits_token_in_input_by_target_by_molecule, \
            max_tokens_by_molecule, \
            offset_token_by_target_by_molecule, \
            shape_token_by_statement_by_molecule, \
            shape_token_by_statement_by_role_by_molecule = get_target_and_statement_token_ids(
                molecules=molecules,
                statements=statements,
                targets=targets,
                max_length_by_molecule=max_length_by_molecule,
                n_molecules=n_molecules,
                target_offset_by_molecule=target_offset_by_molecule,
                max_targets_by_molecule=max_targets_by_molecule
            )
            statement_offset_by_molecule = offset_token_by_target_by_molecule.max(axis=(1, 2)) + 1

            if text_pair is not None:
                max_targets_by_molecule_pair = molecules_pair['n_targets'].max()
                target_offset_by_molecule_pair = statement_offset_by_molecule + \
                                                 fits_token_in_input_by_statement_by_molecule.sum((1, 2)) + 1

                input_ids_by_target_by_molecule_pair, \
                input_ids_by_statement_by_role_by_molecule_pair, \
                is_statement_in_input_by_molecule_pair, \
                fits_token_in_input_by_statement_by_molecule_pair, \
                fits_token_in_input_by_statement_by_role_by_molecule_pair, \
                fits_token_in_input_by_target_by_molecule_pair, \
                max_tokens_by_molecule_pair, \
                offset_token_by_target_by_molecule_pair, \
                shape_token_by_statement_by_molecule_pair, \
                shape_token_by_statement_by_role_by_molecule_pair = get_target_and_statement_token_ids(
                    molecules=molecules_pair,
                    statements=statements_pair,
                    targets=targets_pair,
                    max_length_by_molecule=max_length_by_molecule_pair,
                    n_molecules=n_molecules,
                    target_offset_by_molecule=target_offset_by_molecule_pair,
                    max_targets_by_molecule=max_targets_by_molecule_pair
                )
                statement_offset_by_molecule_pair = offset_token_by_target_by_molecule_pair.max(axis=(1, 2)) + 1

            # initialize input id tensor
            shape_molecules_by_max_seq_length = (n_molecules, max_length)
            input_ids_batch = np.zeros(shape_molecules_by_max_seq_length, dtype=int)

            # fill in cls token ids
            cls_tokens_y = np.arange(n_molecules)
            cls_token_offset_by_molecule = np.zeros(n_molecules, dtype=int)
            cls_tokens_x = cls_token_offset_by_molecule
            input_ids_batch[cls_tokens_y, cls_tokens_x] = self.base_tokenizer.cls_token_id

            # fill in molecule token ids
            n_statement_tokens_by_molecule, \
            n_target_tokens_by_molecule, \
            offset_token_by_statement_by_role_by_molecule, \
            x_target_tokens, \
            y_molecule_4_target_tokens = fill_in_molecule_input_ids(
                fits_token_in_input_by_statement_by_role_by_molecule,
                fits_token_in_input_by_target_by_molecule,
                input_ids_batch,
                input_ids_by_statement_by_role_by_molecule,
                input_ids_by_target_by_molecule,
                max_tokens_by_molecule,
                n_molecules,
                offset_token_by_target_by_molecule,
                shape_token_by_statement_by_role_by_molecule,
                statement_offset_by_molecule
            )

            if text_pair is not None:
                # fill in SEP token ids
                sep_token_offset_by_molecule = target_offset_by_molecule_pair - 1
                molecule_range = np.arange(n_molecules)
                input_ids_batch[molecule_range, sep_token_offset_by_molecule] = self.base_tokenizer.sep_token_id

                # fill in input ids of second molecule
                n_statement_tokens_by_molecule_pair, \
                n_target_tokens_by_molecule_pair, \
                offset_token_by_statement_by_role_by_molecule_pair, \
                x_target_tokens_pair, \
                y_molecule_4_target_tokens_pair = fill_in_molecule_input_ids(
                    fits_token_in_input_by_statement_by_role_by_molecule_pair,
                    fits_token_in_input_by_target_by_molecule_pair,
                    input_ids_batch,
                    input_ids_by_statement_by_role_by_molecule_pair,
                    input_ids_by_target_by_molecule_pair,
                    max_tokens_by_molecule_pair,
                    n_molecules,
                    offset_token_by_target_by_molecule_pair,
                    shape_token_by_statement_by_role_by_molecule_pair,
                    statement_offset_by_molecule_pair
                )

            # Compute position ids
            position_ids_batch = np.zeros(shape_molecules_by_max_seq_length, dtype=int)
            position_offset_first_token_by_molecule = np.repeat(1, n_molecules)

            fill_in_molecule_position_ids(
                fits_token_in_input_by_statement_by_role_by_molecule=fits_token_in_input_by_statement_by_role_by_molecule,
                fits_token_in_input_by_target_by_molecule=fits_token_in_input_by_target_by_molecule,
                offset_token_by_statement_by_role_by_molecule=offset_token_by_statement_by_role_by_molecule,
                position_ids=position_ids_batch,
                x_target_tokens=x_target_tokens,
                y_molecule_4_target_tokens=y_molecule_4_target_tokens,
                position_offset_first_token_by_molecule=position_offset_first_token_by_molecule,
            )

            if text_pair is not None:
                # fill in position ids of sep tokens
                position_offset_sep_token_by_molecule = position_ids_batch.max(1) + 1
                position_ids_batch[molecule_range, sep_token_offset_by_molecule] = position_offset_sep_token_by_molecule
                position_offset_first_token_by_molecule_pair = position_offset_sep_token_by_molecule + 1

                fill_in_molecule_position_ids(
                    fits_token_in_input_by_statement_by_role_by_molecule=fits_token_in_input_by_statement_by_role_by_molecule_pair,
                    fits_token_in_input_by_target_by_molecule=fits_token_in_input_by_target_by_molecule_pair,
                    offset_token_by_statement_by_role_by_molecule=offset_token_by_statement_by_role_by_molecule_pair,
                    position_ids=position_ids_batch,
                    x_target_tokens=x_target_tokens_pair,
                    y_molecule_4_target_tokens=y_molecule_4_target_tokens_pair,
                    position_offset_first_token_by_molecule=position_offset_first_token_by_molecule_pair
                )
            if self.tm_attention:
                # initialize attention masks
                attention_masks_batch = self.get_attention_masks(
                    cls_token_offset_by_molecule,
                    fits_token_in_input_by_statement_by_molecule,
                    fits_token_in_input_by_statement_by_molecule_pair,
                    fits_token_in_input_by_target_by_molecule,
                    fits_token_in_input_by_target_by_molecule_pair,
                    is_statement_in_input_by_molecule,
                    is_statement_in_input_by_molecule_pair,
                    max_length,
                    max_length_by_molecule,
                    max_length_by_molecule_pair,
                    molecule_range,
                    n_molecules,
                    n_statement_tokens_by_molecule,
                    n_statement_tokens_by_molecule_pair,
                    n_target_tokens_by_molecule,
                    n_target_tokens_by_molecule_pair,
                    offset_token_by_statement_by_role_by_molecule,
                    offset_token_by_statement_by_role_by_molecule_pair,
                    offset_token_by_target_by_molecule,
                    offset_token_by_target_by_molecule_pair,
                    sep_token_offset_by_molecule,
                    shape_token_by_statement_by_molecule,
                    shape_token_by_statement_by_molecule_pair,
                    statement_offset_by_molecule,
                    statement_offset_by_molecule_pair,
                    target_offset_by_molecule,
                    target_offset_by_molecule_pair,
                    text_pair)
            if text_pair is None:
                # compute targets mask
                targets_mask_batch = np.zeros((n_molecules, max_targets_by_molecule, max_length))
                z_molecule_4_targets_4_tokens = np.concatenate([
                    np.repeat(i, n_target_tokens) for i, n_target_tokens in enumerate(n_target_tokens_by_molecule)
                ])
                n_tokens_by_target_by_molecule = fits_token_in_input_by_target_by_molecule.sum(2)
                y_target_4_tokens = np.concatenate([
                    np.repeat(i, n_tokens)
                    for n_tokens_by_target in n_tokens_by_target_by_molecule
                    for i, n_tokens in enumerate(n_tokens_by_target)
                ])
                x_tokens = np.concatenate(
                    [np.arange(1, n_target_tokens + 1) for n_target_tokens in n_target_tokens_by_molecule])
                targets_mask_batch[z_molecule_4_targets_4_tokens, y_target_4_tokens, x_tokens] = 1
                targets_mask.append(targets_mask_batch)
            else:
                n_tokens_by_molecule_pair = n_target_tokens_by_molecule_pair + n_statement_tokens_by_molecule_pair
                token_type_ids_batch = np.zeros(shape_molecules_by_max_seq_length)
                type_id_holes_y = np.concatenate([np.repeat(k, v) for k, v in enumerate(n_tokens_by_molecule_pair)])
                type_id_holes_x = np.concatenate([np.arange(offset, offset + n_tokens) for offset, n_tokens in zip(sep_token_offset_by_molecule + 1, n_tokens_by_molecule_pair)])
                token_type_ids_batch[type_id_holes_y, type_id_holes_x] = 1
                token_type_ids.append(token_type_ids_batch)
            input_ids.append(input_ids_batch)
            position_ids.append(position_ids_batch)
            if self.tm_attention:
                attention_masks.append(attention_masks_batch)

        encoding_data = {
            'input_ids': torch.IntTensor(np.concatenate(input_ids)),
            'position_ids': torch.IntTensor(np.concatenate(position_ids)),
        }
        if text_pair is None:
            encoding_data['targets_mask'] = torch.IntTensor(np.concatenate(targets_mask))
        else:
            encoding_data['token_type_ids'] = torch.IntTensor(np.concatenate(token_type_ids))
        if self.tm_attention:
            encoding_data['attention_mask'] = torch.IntTensor(np.concatenate(attention_masks))

        return BatchEncoding(data=encoding_data)

    def get_attention_masks(self, cls_token_offset_by_molecule, fits_token_in_input_by_statement_by_molecule,
                            fits_token_in_input_by_statement_by_molecule_pair,
                            fits_token_in_input_by_target_by_molecule, fits_token_in_input_by_target_by_molecule_pair,
                            is_statement_in_input_by_molecule, is_statement_in_input_by_molecule_pair, max_length,
                            max_length_by_molecule, max_length_by_molecule_pair, molecule_range, n_molecules,
                            n_statement_tokens_by_molecule, n_statement_tokens_by_molecule_pair,
                            n_target_tokens_by_molecule, n_target_tokens_by_molecule_pair,
                            offset_token_by_statement_by_role_by_molecule,
                            offset_token_by_statement_by_role_by_molecule_pair, offset_token_by_target_by_molecule,
                            offset_token_by_target_by_molecule_pair, sep_token_offset_by_molecule,
                            shape_token_by_statement_by_molecule, shape_token_by_statement_by_molecule_pair,
                            statement_offset_by_molecule, statement_offset_by_molecule_pair, target_offset_by_molecule,
                            target_offset_by_molecule_pair, text_pair):
        attention_masks_batch = np.zeros((n_molecules, *2 * [max_length]), dtype=int)
        # Add holes for cls token
        seq_length_by_molecule = add_molecule_seeing_token_holes(
            attention_masks_batch,
            offset_token_by_statement_by_role_by_molecule,
            offset_token_by_target_by_molecule,
            target_offset_by_molecule,
            token_offset_by_molecule=cls_token_offset_by_molecule
        )
        # cls tokens can see themselves
        attention_masks_batch[:, 0, 0] = 1
        if text_pair is not None:
            # add holes for cls token for second molecules
            seq_length_by_molecule_pair = add_molecule_seeing_token_holes(
                attention_masks_batch,
                offset_token_by_statement_by_role_by_molecule_pair,
                offset_token_by_target_by_molecule_pair,
                target_offset_by_molecule_pair,
                token_offset_by_molecule=cls_token_offset_by_molecule
            )
            # add holes for sep token
            for inputs in [
                (offset_token_by_statement_by_role_by_molecule,
                 offset_token_by_target_by_molecule,
                 target_offset_by_molecule),
                (offset_token_by_statement_by_role_by_molecule_pair,
                 offset_token_by_target_by_molecule_pair,
                 target_offset_by_molecule_pair)
            ]:
                _ = add_molecule_seeing_token_holes(
                    attention_masks_batch,
                    *inputs,
                    token_offset_by_molecule=sep_token_offset_by_molecule
                )
            # sep tokens can see themselves
            attention_masks_batch[molecule_range, sep_token_offset_by_molecule, sep_token_offset_by_molecule] = 1
            # sep and cls tokens can see each other
            attention_masks_batch[molecule_range, sep_token_offset_by_molecule, 0] = 1
            attention_masks_batch[molecule_range, 0, sep_token_offset_by_molecule] = 1

        # add holes for molecule tokens
        n_tokens_by_target_by_molecule = add_molecule_holes(
            attention_masks_batch,
            is_statement_in_input_by_molecule,
            fits_token_in_input_by_statement_by_molecule,
            fits_token_in_input_by_target_by_molecule,
            max_length_by_molecule,
            n_statement_tokens_by_molecule,
            n_target_tokens_by_molecule,
            offset_token_by_statement_by_role_by_molecule,
            offset_token_by_target_by_molecule,
            seq_length_by_molecule,
            shape_token_by_statement_by_molecule,
            target_offset_by_molecule,
            statement_offset_by_molecule
        )
        if text_pair is not None:
            # add holes for targets and statements of molecule pairs
            n_tokens_by_target_by_molecule_pair = add_molecule_holes(
                attention_masks_batch,
                is_statement_in_input_by_molecule_pair,
                fits_token_in_input_by_statement_by_molecule_pair,
                fits_token_in_input_by_target_by_molecule_pair,
                max_length_by_molecule_pair,
                n_statement_tokens_by_molecule_pair,
                n_target_tokens_by_molecule_pair,
                offset_token_by_statement_by_role_by_molecule_pair,
                offset_token_by_target_by_molecule_pair,
                seq_length_by_molecule_pair,
                shape_token_by_statement_by_molecule_pair,
                target_offset_by_molecule_pair,
                statement_offset_by_molecule_pair
            )
            # add holes between molecules for targets (targets from molecules and targets from molecule pairs can see
            # each other)
            product_target_tokens_by_molecule = n_target_tokens_by_molecule * n_target_tokens_by_molecule_pair
            z_molecule_4_tokens_4_tokens = np.concatenate([np.repeat(i, product_target_tokens)
                                                           for i, product_target_tokens
                                                           in enumerate(product_target_tokens_by_molecule)])
            y_token_4_tokens = np.concatenate([
                np.repeat(np.arange(target_offset_pair, n_target_tokens_pair + target_offset_pair), n_target_tokens)
                for target_offset_pair, n_target_tokens_pair, n_target_tokens
                in
                zip(target_offset_by_molecule_pair, n_target_tokens_by_molecule_pair, n_target_tokens_by_molecule)
            ])
            x_tokens = np.concatenate([
                np.tile(np.arange(1, n_target_tokens + 1), n_target_tokens_pair)
                for n_target_tokens_pair, n_target_tokens
                in zip(n_target_tokens_by_molecule_pair, n_target_tokens_by_molecule)
            ])
            attention_masks_batch[z_molecule_4_tokens_4_tokens, y_token_4_tokens, x_tokens] = 1
            attention_masks_batch[z_molecule_4_tokens_4_tokens, x_tokens, y_token_4_tokens] = 1
        return attention_masks_batch

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

    def pad(
            self,
            encoded_inputs: Union[
                BatchEncoding,
                List[BatchEncoding],
                Dict[str, EncodedInput],
                Dict[str, List[EncodedInput]],
                List[Dict[str, EncodedInput]],
            ],
            padding: Union[bool, str, PaddingStrategy] = True,
            max_length: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            verbose: bool = True
    ):
        return self.base_tokenizer.pad(encoded_inputs)

    def save_pretrained(self, *args, **kwargs):
        self.base_tokenizer.save_pretrained(*args, **kwargs)

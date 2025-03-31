# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union, Optional, Any
import copy
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        obj_array = np.empty(len(val), dtype=object)
        obj_array[:] = val
        non_tensors[key] = obj_array

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)


    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                # XXX: consider multiturn
                tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                                 axis=1)]

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')


    @staticmethod
    def get_actual_userturn_index(messages: list[dict]) -> int:
        """
        Returns the index of the first message in the last contiguous block of user messages.
        If the last message is not a user message, returns -1.

        In a multi-turn conversation, a user may send multiple consecutive messages
        (e.g. user1, assistant, user2, user3). The function returns the index of the
        first message of the last contiguous block of user messages (user2 in the example).
        """
        assert len(messages) > 0, "Messages should not be empty"
        assert messages[-1]["role"] == "user", "Last message should be a user message"

        # Search backward to find the last user message
        i = len(messages) - 1
        while i > 0 and messages[i - 1]["role"] == "user":
            i -= 1
        return i

    def try_create_multiturn_data(
        self,
        data_index: int,
        response: str,
        raw_prompt: list[dict] | np.ndarray,
        remaining_prompt: list[dict] | np.ndarray,
    ) -> dict:
        """
        If adding the response won't cause the dialogs to finish, we form a new dialog.
        """
        assert len(remaining_prompt) > 0, "Remaining prompt should not be empty"
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if isinstance(remaining_prompt, np.ndarray):
            remaining_prompt = remaining_prompt.tolist()
        raw_data = self.dataframe.iloc[data_index].to_dict()
        new_message = dict(role="assistant", content=response)
        new_messages = raw_prompt + [new_message] + remaining_prompt
        raw_data[self.prompt_key] = np.array(new_messages, dtype=object)
        return raw_data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        return self.process_row(item, row_dict)

    def process_row(self, index: int, row_dict: dict) -> dict:
        # Let's store the data index to help multiturn construction
        row_dict["data_index"] = index
        chat = row_dict.pop(self.prompt_key)
        if isinstance(chat, np.ndarray):
            chat = chat.tolist()
        actual_userturn_index = self.get_actual_userturn_index(chat)
        # Remaining multiturn chat
        remaining_chat = chat[actual_userturn_index + 1 :]
        actual_chat = chat[: actual_userturn_index + 1]

        prompt_with_chat_template = self.tokenizer.apply_chat_template(actual_chat, add_generation_prompt=True, tokenize=False)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['remaining_prompt'] = remaining_chat
            row_dict['raw_prompt'] = actual_chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()


BatchDictType = dict[str, list | torch.Tensor]


@dataclass
class MultiTurnPromptManager:
    """
    Maintains:
        1) A queue of multi-turn prompts that still need to be continued in future batches.
        2) A cache of single-turn prompts that can be re-used for partially filling the batch.
    """

    # For incomplete multi-turn prompts from previous steps
    multiturn_queue: deque[dict] = field(default_factory=deque)
    # For storing single-turn prompts that have been replaced in prior steps
    singleturn_cache: deque[dict] = field(default_factory=deque)
    mandatory_batch_key: str = "input_ids"

    def replace_prompts_inplace(self, batch_dict: BatchDictType) -> None:
        """
        1) If we have N multi-turn prompts in the queue, then we replace the LAST min(N, batch_size)
           items in `batch_dict` with the first min(N, batch_size) items from `multiturn_queue`.
           - Also, we take those replaced batch items and store them in `singleturn_cache`.
        2) For the remaining front of the batch, we replace up to that many from `singleturn_cache`.
           - Again, replaced original items are stored back into `singleturn_cache`.
        3) Return the modified `batch_dict`.

        NOTE: This assumes `batch_dict` has lists or tensors of the same length (batch_size).
        """

        batch_size = len(batch_dict[self.mandatory_batch_key])
        # --------------
        # Step 1: Replace the FIRST K = min(len(multiturn_queue), batch_size) items with multi-turn prompts
        # --------------

        K = min(len(self.multiturn_queue), batch_size)

        # Prepare for step 2
        remaining_replacements = batch_size - K
        L = min(len(self.singleturn_cache), remaining_replacements)

        # We'll operate on indices from (batch_size - K) to (batch_size-1)
        for idx in range(K):
            # Pop an item from multi-turn queue
            multiturn_item = self.multiturn_queue.popleft()

            # Save original batch item to single-turn cache
            original_item = self._extract_batch_item(batch_dict, idx)
            # breakpoint()
            self.singleturn_cache.append(original_item)

            # Insert the multi-turn item into the batch
            self._put_batch_item(batch_dict, idx, multiturn_item)

        # --------------
        # Step 2: For the NEXT L items, try replacing them from singleturn_cache
        # --------------

        # Replace the FIRST L items in the batch
        for idx in range(K, K + L):
            singleturn_item = self.singleturn_cache.popleft()

            # Save original batch item to single-turn cache
            original_item = self._extract_batch_item(batch_dict, idx)
            self.singleturn_cache.append(original_item)

            # Insert the single-turn item into the batch
            self._put_batch_item(batch_dict, idx, singleturn_item)


    def update_multiturn_prompts(self, new_prompts: list[dict[str, Any]]) -> None:
        """
        Append new incomplete multi-turn prompts to the queue for use in future batches.
        Typically called after generation steps when we identify incomplete dialogues.
        """
        for prompt in new_prompts:
            self.multiturn_queue.append(prompt)

    @staticmethod
    def _extract_batch_item(batch_dict: BatchDictType, idx: int) -> dict[str, Any]:
        """
        Take a snapshot of the 'idx'-th item from each list/tensor in batch_dict.
        Return it as a dictionary that can be re-injected later.
        """
        return {key: val_list[idx] for key, val_list in batch_dict.items()}

    @staticmethod
    def _put_batch_item(
        batch_dict: BatchDictType, idx: int, item: dict[str, Any]
    ) -> None:
        """
        Place the given 'item' into the 'idx'-th position of each list/tensor in batch_dict.
        """
        for key in batch_dict.keys():
            if isinstance(batch_dict[key], torch.Tensor):
                # Rebuild the entire tensor
                # Otherwise, the extracted batch item may be affected by the change
                # as that's just a view of the original tensor.
                old_shape = batch_dict[key].shape  # type: ignore
                new_tensor = torch.stack(
                    [
                        item[key] if i == idx else batch_dict[key][i]
                        for i in range(old_shape[0])
                    ]
                )
                assert new_tensor.shape == old_shape
                batch_dict[key] = new_tensor
            else:
                # Replace the list item
                batch_dict[key][idx] = item[key]

    def queue_status(self) -> str:
        """
        Return the number of items in the multiturn_queue and singleturn_cache.
        """
        return f"Multiturn queue: {len(self.multiturn_queue)}, Singleturn cache: {len(self.singleturn_cache)}"

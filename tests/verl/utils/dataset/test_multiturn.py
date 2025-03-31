from verl.utils.dataset.rl_dataset import (
    RLHFDataset,
    collate_fn,
    MultiTurnPromptManager,
)
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto
from verl.utils import hf_tokenizer
import tempfile
import pandas as pd

multiturn_prompt = [
    dict(role="user", content="What are you doing?"),
    dict(role="user", content="Any luck?"),
]

mock_data = [dict(prompt=multiturn_prompt)]
# Save it to a temporary parquet file

with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
    pd.DataFrame(mock_data).to_parquet(f.name)
    f.flush()
    print("Saving mock data to", f.name)

    tokenizer = hf_tokenizer("Qwen/Qwen2.5-3B-Instruct")
    dataset = RLHFDataset(
        parquet_files=f.name,
        tokenizer=tokenizer,
        processor=None,
        prompt_key="prompt",
        # image_key=self.config.data.get('image_key', 'images'),
        max_prompt_length=3072,
        filter_prompts=True,
        return_raw_chat=True,
        # truncation=
        # filter_overlong_prompts=self.config.data.filter_overlong_prompts
    )

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=1,  # hardcode the 2x
        num_workers=1,
        drop_last=True,
        collate_fn=collate_fn,
        # sampler=sampler
    )

    prompt_manager = MultiTurnPromptManager()

    batch_dict = next(iter(dataloader))
    batch: DataProto = DataProto.from_dict(batch_dict)
    old_prompt = batch_dict["input_ids"][0]

    data_indices = batch.batch["data_index"]

    # example_multi = copy.deepcopy(dataset.dataframe.iloc[0].to_dict())
    newdata = dataset.try_create_multiturn_data(
        0, "1 or 2", batch_dict["raw_prompt"][0], batch_dict["remaining_prompt"][0]
    )
    row_processed = dataset.process_row(0, newdata)

    prompt_manager.update_multiturn_prompts([row_processed])
    prompt_manager.replace_prompts_inplace(batch_dict)

    new_prompt = batch_dict["input_ids"][0]

    print("OLD PROMPT:", tokenizer.decode(old_prompt, skip_special_tokens=True), sep="\n")
    print("NEW PROMPT:", tokenizer.decode(new_prompt, skip_special_tokens=True), sep="\n")

"""Direct Generation Inferencer."""

import inspect
import json
import os
import os.path as osp
import time
from typing import List, Optional

import mmengine
import torch
from tqdm import tqdm

from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS
from opencompass.utils import batched

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class GenInferencer(BaseInferencer):
    """Generation Inferencer class to directly evaluate by generation.

    Attributes:
        model (:obj:`BaseModelWrapper`, optional): The module to inference.
        max_seq_len (:obj:`int`, optional): Maximum number of tokenized words
            allowed by the LM.
        min_out_len (:obj:`int`, optional): Minimum number of generated tokens
            by the LM
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the
            generation field token when generating prompts.
        save_every (:obj:`int`, optional): Save intermediate results every
            `save_every` iters. Defaults to 1.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the
            :obj:`model.generate()` method.
    """

    def __init__(
            self,
            model: BaseModel,
            max_out_len: int,
            stopping_criteria: List[str] = [],
            max_seq_len: Optional[int] = None,
            min_out_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            gen_field_replace_token: Optional[str] = '',
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )

        self.gen_field_replace_token = gen_field_replace_token
        self.max_out_len = max_out_len
        self.min_out_len = min_out_len
        self.stopping_criteria = stopping_criteria
        self.dump_timer = kwargs.get('dump_timer', False)

        if self.model.is_api and save_every is None:
            save_every = 1
        self.save_every = save_every

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        # 3.1 Fetch and zip prompt & gold answer if output column exists
        ds_reader = retriever.dataset_reader
        if ds_reader.output_column:
            gold_ans = ds_reader.dataset['test'][ds_reader.output_column]
            prompt_list = list(zip(prompt_list, gold_ans))

        # Create tmp json file for saving intermediate results and future
        # resuming
        index = 0
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)
        if osp.exists(tmp_json_filepath):
            # TODO: move resume to output handler
            try:
                tmp_result_dict = mmengine.load(tmp_json_filepath)
            except Exception:
                pass
            else:
                output_handler.results_dict = tmp_result_dict
                index = len(tmp_result_dict)

        # 4. Wrap prompts with Dataloader
        logger.info('Starting build dataloader')
        dataloader = self.get_dataloader(prompt_list[index:], self.batch_size)

        # 5. Inference for prompts in each batch
        logger.info('Starting inference process...')

        # Calibration phase for accurate ETA
        num_remaining_samples = len(prompt_list) - index
        num_batches = len(dataloader)
        calibration_batches = 1  # Show ETA after just 1 batch!

        if num_batches > 0 and calibration_batches > 0:
            logger.info(f'Running calibration on first batch to measure speed...')

        start_time_stamp = time.time()
        num_sample = 0
        batch_num = 0
        calibration_done = False

        for datum in tqdm(dataloader, disable=not self.is_main_process):
            batch_start = time.time()
            batch_num += 1
            if ds_reader.output_column:
                entry, golds = list(zip(*datum))
            else:
                entry = datum
                golds = [None for _ in range(len(entry))]
            # 5-1. Inference with local model
            extra_gen_kwargs = {}
            sig = inspect.signature(self.model.generate)
            if 'stopping_criteria' in sig.parameters:
                extra_gen_kwargs['stopping_criteria'] = self.stopping_criteria
            if 'min_out_len' in sig.parameters:
                extra_gen_kwargs['min_out_len'] = self.min_out_len
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entry, mode='gen')
                results = self.model.generate_from_template(
                    entry, max_out_len=self.max_out_len, **extra_gen_kwargs)
                generated = results

            num_return_sequences = getattr(self.model, 'generation_kwargs',
                                           {}).get('num_return_sequences', 1)
            # 5-3. Save current output
            for prompt, prediction, gold in zip(
                    parsed_entries, batched(generated, num_return_sequences),
                    golds):
                if num_return_sequences == 1:
                    prediction = prediction[0]
                output_handler.save_results(prompt,
                                            prediction,
                                            index,
                                            gold=gold)
                index = index + 1

            # 5-4. Save intermediate results
            if (self.save_every is not None and index % self.save_every == 0
                    and self.is_main_process):
                output_handler.write_to_json(output_json_filepath,
                                             'tmp_' + output_json_filename)
            num_sample += len(datum)

            # After first batch, show total ETA immediately
            if not calibration_done and batch_num >= calibration_batches:
                calibration_done = True
                batch_time = time.time() - batch_start  # Time for this batch
                remaining_batches = num_batches - batch_num
                estimated_remaining_seconds = remaining_batches * batch_time
                total_estimated_seconds = (time.time() - start_time_stamp) + estimated_remaining_seconds

                # Format total estimated time
                if total_estimated_seconds < 60:
                    total_time_str = f"{int(total_estimated_seconds)} seconds"
                else:
                    total_minutes = int(total_estimated_seconds / 60)
                    total_seconds = int(total_estimated_seconds % 60)
                    total_time_str = f"{total_minutes}m {total_seconds}s"

                logger.info(f'Speed measured: {batch_time:.1f}s per batch (based on first batch).')
                logger.info(f'📊 ESTIMATED TIME FOR THIS TASK: {total_time_str} ({num_batches} batches, {num_remaining_samples} samples)')

                # Check if this is the first task and calculate TOTAL evaluation ETA
                try:
                    # Try to find the eta_metadata file in work_dir/tmp/
                    # output_json_filepath is like: work_dir/predictions/model/dataset
                    # Split by 'predictions' to get work_dir
                    if 'predictions' in output_json_filepath:
                        work_dir_path = output_json_filepath.split('predictions')[0].rstrip('/')
                    else:
                        work_dir_path = os.path.dirname(os.path.dirname(output_json_filepath))
                    eta_metadata_file = os.path.join(work_dir_path, 'tmp', 'eta_metadata.json')

                    if os.path.exists(eta_metadata_file):
                        with open(eta_metadata_file, 'r') as f:
                            metadata = json.load(f)

                        if not metadata.get('calibration_complete', False):
                            # This is the first task to calibrate!
                            total_batches = metadata['total_batches']
                            total_tasks = metadata['total_tasks']

                            # Calculate TOTAL evaluation time using measured speed
                            total_eval_time = total_batches * batch_time

                            # Format total evaluation time
                            if total_eval_time < 60:
                                total_eval_time_str = f"{int(total_eval_time)}s"
                            elif total_eval_time < 3600:
                                total_eval_minutes = int(total_eval_time / 60)
                                total_eval_seconds = int(total_eval_time % 60)
                                total_eval_time_str = f"{total_eval_minutes}m {total_eval_seconds}s"
                            else:
                                total_eval_hours = int(total_eval_time / 3600)
                                remaining_minutes = int((total_eval_time % 3600) / 60)
                                total_eval_time_str = f"{total_eval_hours}h {remaining_minutes}m"

                            logger.info(f'')
                            logger.info(f'📊 TOTAL EVALUATION ETA: {total_eval_time_str} '
                                       f'({total_batches} batches across {total_tasks} tasks)')

                            # Add machine-parseable JSON log for programmatic extraction
                            eta_data = {
                                "total_eta_seconds": int(total_eval_time),
                                "total_batches": total_batches,
                                "total_tasks": total_tasks,
                                "speed_per_batch": round(batch_time, 2)
                            }
                            logger.info(f'ETA_DATA: {json.dumps(eta_data)}')
                            logger.info(f'')

                            # Mark calibration as complete
                            metadata['calibration_complete'] = True
                            metadata['speed_per_batch'] = batch_time
                            with open(eta_metadata_file, 'w') as f:
                                json.dump(metadata, f)
                except Exception:
                    # Silently ignore if metadata file doesn't exist or can't be read
                    pass

        end_time_stamp = time.time()

        # Log completion summary
        total_time = end_time_stamp - start_time_stamp
        if total_time < 60:
            time_str = f"{total_time:.1f}s"
        else:
            minutes = int(total_time / 60)
            seconds = int(total_time % 60)
            time_str = f"{minutes}m {seconds}s"

        if num_sample > 0:
            avg_time = total_time / num_sample
            logger.info(f'Inference complete! Processed {num_sample} samples in '
                       f'{time_str} (avg {avg_time:.2f}s/sample)')
        else:
            logger.info(f'Inference complete! Total time: {time_str}')

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        if self.dump_timer and self.is_main_process:
            timer_filepath = os.path.join(output_json_filepath, 'timer',
                                          'time.jsonl')
            os.makedirs(os.path.dirname(timer_filepath), exist_ok=True)
            time_dict = {
                'dataset_name': output_json_filename.removesuffix('.json'),
                'time': end_time_stamp - start_time_stamp,
                'num_sample': num_sample
            }
            with open(timer_filepath, 'a') as f:
                f.write(json.dumps(time_dict) + '\n')

        return [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]

    def get_generation_prompt_list_from_retriever_indices(
            self,
            ice_idx_list: List[List[int]],
            retriever: BaseRetriever,
            gen_field_replace_token: str,
            max_seq_len: Optional[int] = None,
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None):
        prompt_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=gen_field_replace_token,
                ice_template=ice_template,
                prompt_template=prompt_template)
            if max_seq_len is not None:
                prompt_token_num = self.model.get_token_len_from_template(
                    prompt, mode='gen')
                while len(ice_idx) > 0 and prompt_token_num > max_seq_len:
                    ice_idx = ice_idx[:-1]
                    ice = retriever.generate_ice(ice_idx,
                                                 ice_template=ice_template)
                    prompt = retriever.generate_prompt_for_generate_task(
                        idx,
                        ice,
                        gen_field_replace_token=gen_field_replace_token,
                        ice_template=ice_template,
                        prompt_template=prompt_template)
                    prompt_token_num = self.model.get_token_len_from_template(
                        prompt, mode='gen')
            prompt_list.append(prompt)
        return prompt_list


@ICL_INFERENCERS.register_module()
class GLMChoiceInferencer(GenInferencer):

    def __init__(self, *args, choices=['A', 'B', 'C', 'D'], **kwargs):
        super().__init__(*args, **kwargs)
        self.choices = choices

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        # 4. Wrap prompts with Dataloader
        dataloader = self.get_dataloader(prompt_list, self.batch_size)
        index = 0

        # 5. Inference for prompts in each batch
        logger.info('Starting inference process...')
        for entry in tqdm(dataloader, disable=not self.is_main_process):
            # 5-1. Inference with local model
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entry, mode='gen')
                results = self.model.choice(entry, choices=self.choices)
                generated = results

            # 5-3. Save current output
            for prompt, prediction in zip(parsed_entries, generated):
                output_handler.save_results(prompt, prediction, index)
                index = index + 1

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
        return [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]

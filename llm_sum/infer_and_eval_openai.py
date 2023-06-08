import argparse
import csv
import logging
import random
import time

import openai
from rouge_score import rouge_scorer, scoring
import tiktoken

_MODEL_MAX_TOKENS = 4096

_LOGGING_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_test_data_id',
        type=int,
        default=0,
        help='Start test data id.',
    )
    parser.add_argument(
        '--output_csv_path',
        type=str,
        default='output.csv',
        help='Path to output csv file.',
    )
    parser.add_argument(
        '--logging_level',
        type=str,
        default='warning',
        choices=['debug', 'info', 'warning'],
        help='Logging level.',
    )
    return parser.parse_args()


def get_sentence_by_line_number(source_path: str, line_number: int) -> str:
    with open(source_path, 'r') as f:
        source = f.readlines()
    return source[line_number].strip()


def calc_rouge(
    scorer: rouge_scorer.RougeScorer,
    reference: str,
    generated: str,
) -> None:
    scores = scorer.score(reference, generated)
    logging.info(
        '\n'
        f'rouge1: precision={scores["rouge1"].precision:.4f}, '
        f'recall={scores["rouge1"].recall:.4f}, '
        f'f1={scores["rouge1"].fmeasure:.4f},\n'
        f'rouge2: precision={scores["rouge2"].precision:.4f}, '
        f'recall={scores["rouge2"].recall:.4f}, '
        f'f1={scores["rouge2"].fmeasure:.4f},\n'
        f'rougeL: precision={scores["rougeL"].precision:.4f}, '
        f'recall={scores["rougeL"].recall:.4f}, '
        f'f1={scores["rougeL"].fmeasure:.4f},\n'
    )


def set_openai_credentials(
    api_key_path: str = '.credentials/api_key',
    organization_path: str = '.credentials/organization',
) -> None:
    with open(api_key_path, 'r') as f:
        openai.api_key = f.read()
    with open(organization_path, 'r') as f:
        openai.organization = f.read()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # note: future models may deviate from this
    if model == "gpt-3.5-turbo-0301":
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                # if there's a name, the role is omitted
                if key == "name":
                    # role is always required and always 1 token
                    num_tokens += -1
        # every reply is primed with <im_start>assistant
        num_tokens += 2

        # Note(yujisw): I do not know why, but num_tokens needs to be
        # increased by 1 to get the correct number of tokens.
        return num_tokens+1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented
            for model {model}. See
            https://github.com/openai/openai-python/blob/main/chatml.md
            for information on how messages are converted to tokens."""
        )


def create_prompt(
    source_test: str,
    few_shot: bool,
    specify_length: int = None,
    use_task_description: bool = True,
    example_source_text: str = None,
    example_summary_text: str = None,
) -> list[dict[str, str]]:
    if few_shot and (
        example_source_text is None
        or example_summary_text is None
    ):
        raise ValueError(
            'Specify example_source_text and example_summary_text '
            'in few-shot setting.'
        )

    system_content_list = []
    role_specification = (
        'You are a good annotator to provide a SUMMARY of a given '
        'SOURCE document.'
    )
    system_content_list.append(role_specification)
    if specify_length is not None:
        length_specification = (
            'The length (number of words) of the summary is specified in the '
            'following format. LENGTH: [word_num].'
        )
        system_content_list.append(length_specification)
    if use_task_description:
        task_description = (
            'The process of creating summaries involves selecting key '
            'information from the source documents and condensing it into '
            'a shorter, concise form. The annotators aim to capture the main '
            'points, important details, and the overall essence of the source '
            'document in the summary.'
        )
        system_content_list.append(task_description)
        if specify_length is not None:
            system_content_list.append(
                'In addition, the annotator is required to keep the length of '
                'the summary to a specified length.'
            )
    if few_shot:
        source_example = f'SOURCE: {example_source_text}'
        summary_example = f'SUMMARY: {example_summary_text}'

        if specify_length is None:
            example_description = (
                'The following is an example of a SOURCE document and a '
                'SUMMARY of the SOURCE document.'
            )
            system_content_list.extend([
                example_description,
                source_example,
                summary_example,
            ])
        else:
            example_description = (
                'The following is an example of a SOURCE document, a '
                'specified LENGTH, and a SUMMARY of the SOURCE document.'
            )
            length_example = f'LENGTH: {len(example_summary_text.split())}'
            system_content_list.extend([
                example_description,
                source_example,
                length_example,
                summary_example,
            ])

    system_content_str = '\n'.join(system_content_list)

    user_content_str = f'SOURCE: {source_test}'
    if specify_length is not None:
        user_content_str += f'\nLENGTH: {specify_length}'

    # calculate the length of the prompt
    prompt = [
        {'role': 'system', 'content': system_content_str},
        {'role': 'user', 'content': user_content_str},
    ]
    logging.debug(f'system:\n{system_content_str}')
    logging.debug(f'user:\n{user_content_str}')

    prompt_length = num_tokens_from_messages(prompt)
    max_tokens = _MODEL_MAX_TOKENS - prompt_length
    logging.info(
        f'The prompt has {prompt_length} tokens.'
        f'So, MAX_TOKENS is {max_tokens}.'
    )
    return system_content_str, user_content_str, prompt_length


def generate_summary(
    test_data_id: int,
    system_content_str: str,
    user_content_str: str,
    prompt_length: int
) -> str:
    for i in range(10):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': system_content_str},
                    {'role': 'user', 'content': user_content_str},
                ],
                temperature=0.0,
                max_tokens=_MODEL_MAX_TOKENS-prompt_length,
            )
        except openai.error.RateLimitError:
            logging.warning(
                f'TEST_DATA_ID: {test_data_id}, '
                f'({i}/10) Rate limit exceeded.'
            )
            time.sleep(10)
        except Exception as e:
            logging.warning(f'TEST_DATA_ID: {test_data_id}, ({i}/10) {e}')
            return None
        else:
            break

    logging.debug(f'{response["usage"]["total_tokens"]} tokens used.')
    if response["choices"][0]["finish_reason"] != "stop":
        logging.warning(
            f'TEST_DATA_ID: {test_data_id}\n'
            f'finish_reason: {response["choices"][0]["finish_reason"]}.\n'
            'The response does not seem to be finished successfully.'
        )

    return response["choices"][0]["message"]["content"]


def _rouge_score_to_list(rouge_score: dict[str, scoring.Score]) -> list[str]:
    return [
        f'''{rouge_score['rouge1'].precision:.4f}''',
        f'''{rouge_score['rouge2'].precision:.4f}''',
        f'''{rouge_score['rougeL'].precision:.4f}''',
        f'''{rouge_score['rouge1'].recall:.4f}''',
        f'''{rouge_score['rouge2'].recall:.4f}''',
        f'''{rouge_score['rougeL'].recall:.4f}''',
        f'''{rouge_score['rouge1'].fmeasure:.4f}''',
        f'''{rouge_score['rouge2'].fmeasure:.4f}''',
        f'''{rouge_score['rougeL'].fmeasure:.4f}''',
    ]


def _main() -> None:
    args = _parse_args()

    logging.basicConfig(level=_LOGGING_LEVEL_DICT[args.logging_level])

    set_openai_credentials(
        api_key_path='.credentials/api_key',
        organization_path='.credentials/organization',
    )

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    # load train data for few-shot prompting
    train_source_path = '../data/cnn_dm/train.source'
    train_target_path = '../data/cnn_dm/train.target'

    train_source_list = open(train_source_path, 'r').readlines()
    train_target_list = open(train_target_path, 'r').readlines()

    # load test data
    test_source_path = '../data/cnn_dm/test.source'
    test_target_path = '../data/cnn_dm/test.target'

    test_source_list = open(test_source_path, 'r').readlines()
    test_target_list = open(test_target_path, 'r').readlines()

    # load generated summaries by sumtopk for comparison
    sumtopk_summary_path = (
        '../output/finetune-proposal-large_20221027163624'
        '/test_cnn_dm.hypo_least_args'
    )
    sumtopk_summary_list = open(sumtopk_summary_path, 'r').readlines()

    train_data_num = len(train_source_list)
    test_data_num = len(test_source_list)

    result_list: list[list[str]] = []

    try:
        for test_data_id in range(args.start_test_data_id, test_data_num):
            if test_data_id % 100 == 0:
                logging.info(f'TEST_DATA_ID: {test_data_id} / {test_data_num}')

            test_source_text = test_source_list[test_data_id].strip()
            test_target_text = test_target_list[test_data_id].strip()

            # length is the number of words
            desired_length = len(test_target_text.split(' '))

            for i in range(100):
                try:
                    example_train_data_id = random.randint(0, train_data_num-1)
                    logging.info(
                        f'train data id randomly chosen: '
                        f'{example_train_data_id}'
                    )

                    example_source_text = train_source_list[
                        example_train_data_id
                    ].strip()
                    example_summary_text = train_target_list[
                        example_train_data_id
                    ].strip()

                    # generate prompt
                    system_content_str, user_content_str, prompt_length = \
                        create_prompt(
                            source_test=test_source_text,
                            specify_length=desired_length,
                            use_task_description=True,
                            few_shot=True,
                            example_source_text=example_source_text,
                            example_summary_text=example_summary_text,
                        )
                    required_length = max(200, desired_length * 2)
                    if required_length > _MODEL_MAX_TOKENS:
                        raise ValueError(
                            f'TEST_DATA_ID: {test_data_id}, '
                            f'prompt_length: {prompt_length}, '
                            f'desired_length: {desired_length}, '
                            f'_MODEL_MAX_TOKENS: {_MODEL_MAX_TOKENS}'
                        )
                except ValueError as e:
                    logging.warning(f'TEST_DATA_ID: {test_data_id}, {e}')
                else:
                    break

            # generate summary
            openai_summary = generate_summary(
                test_data_id,
                system_content_str,
                user_content_str,
                prompt_length
            )

            if openai_summary is None:
                openai_rouge_score_list = [''] * 9
            else:
                # calculate rouge score
                openai_rouge_score = scorer.score(
                    test_target_text,
                    openai_summary
                )
                openai_rouge_score_list = _rouge_score_to_list(
                    openai_rouge_score
                )

            # calculate sumtopk rouge score
            sumtopk_summary = sumtopk_summary_list[test_data_id].strip()
            sumtopk_rouge_score = scorer.score(
                test_target_text,
                sumtopk_summary
            )
            sumtopk_rouge_score_list = _rouge_score_to_list(
                sumtopk_rouge_score
            )

            # append results
            result_list.append(
                [
                    test_data_id,
                    example_train_data_id,
                    system_content_str,
                    user_content_str,
                    openai_summary if openai_summary is not None else '',
                    sumtopk_summary,
                ]
                + openai_rouge_score_list
                + sumtopk_rouge_score_list
            )
    except Exception as e:
        logging.exception(e)
        logging.error(
            f'TEST_DATA_ID: {test_data_id}, {e}\n'
            'An error occurred. Saving results so far.'
        )

    # save results in args.output_csv_path
    with open(args.output_csv_path, 'a') as f:
        writer = csv.writer(f)
        if args.start_test_data_id == 0:
            writer.writerow(
                [
                    'test_data_id',
                    'example_train_data_id',
                    'system_content_str',
                    'user_content_str',
                    'openai_summary',
                    'sumtopk_summary',
                    'openai_rouge1_precision',
                    'openai_rouge2_precision',
                    'openai_rougeL_precision',
                    'openai_rouge1_recall',
                    'openai_rouge2_recall',
                    'openai_rougeL_recall',
                    'openai_rouge1_fmeasure',
                    'openai_rouge2_fmeasure',
                    'openai_rougeL_fmeasure',
                    'sumtopk_rouge1_precision',
                    'sumtopk_rouge2_precision',
                    'sumtopk_rougeL_precision',
                    'sumtopk_rouge1_recall',
                    'sumtopk_rouge2_recall',
                    'sumtopk_rougeL_recall',
                    'sumtopk_rouge1_fmeasure',
                    'sumtopk_rouge2_fmeasure',
                    'sumtopk_rougeL_fmeasure',
                ]
            )
        writer.writerows(result_list)


if __name__ == '__main__':
    _main()

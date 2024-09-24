from dotenv import load_dotenv
import os
import pandas as pd
from Prompts import template, template_v2, reflection
import re
import sys
from benchmark.evaluation.utils.overall_evaluate import evaluate, evaluate_models
import concurrent.futures
from collections import Counter
from openai import OpenAI
from PostProcess import correct_single_template


# Load environment variables from the .env file
load_dotenv(dotenv_path=os.path.join('../../.env'))
try:
    openai = OpenAI(
        base_url="https://router.neutrinoapp.com/api/engines",
        api_key=os.getenv("NEUTRINO_API_KEY"))
    # openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

except Exception as e:
    print("Failed to initialize OpenAI API")

def get_chat_completion(logs, models="gpt-4o", meta_analysis=False):
    logs = f"""\nLogs: {logs} \nResponse:"""
    prompt = template_v2 + logs
    # print('logs:', logs)
    if meta_analysis:
        prompt = reflection + logs
    response = openai.chat.completions.create(
        model=models,
        messages=[
            {"role": "system", "content": "You are an expert in log parsing. Your task is to identify and extract all the dynamic variables in each log message and create static log templates."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return [choice.message.content for choice in response.choices]

def get_chat_completion_mini(logs, model="gpt-4o-mini"):
    logs = f"""\nLogs: {logs} \nResponse: """ # Let's think step by step.
    prompt = template_v2 + logs
    # print('logs:', logs)
    response = openai.chat.completions.create(
        model=model[0],
        messages=[
            {"role": "system", "content": "You are an expert in log parsing. Your task is to identify and extract all the dynamic variables in each log message and create static log templates."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0
    )
    return [response.choices[0].message.content]
def parse_templates(dataset, predictedPath, max_logs=10):
    predictedPathComplete = f'../Iudex/results_new_hyp_scan_wc/{dataset}/{predictedPath}.csv'
    # read csv
    df = pd.read_csv(predictedPathComplete)
    # print number of patterns
    print(f'Number of patterns: {df["PatternID"].nunique()}')
    grouped_df = df.groupby('PatternID')['Content'].apply(lambda x: '\n'.join(x.sample(min(len(x), max_logs), random_state=42)))
    # grouped_df = df.groupby('PatternID')['Content'].apply(lambda x: ' '.join(x.head(max_logs)))
    print(grouped_df)
    # get chat completion for each group'
    templates_dict = {}
    templates_raw = []
    for pattern_id, group in grouped_df.items():
        res = get_chat_completion(group)
        templates_raw.append(res)
        # use regex to replace anything between {} with <*>
        res = re.sub(r'{.*?}', '<*>', res)
        print('template: ', res)
        templates_dict[pattern_id] = res
    # map back to original dataframe
    df['EventTemplate'] = df['PatternID'].map(templates_dict)
    print(df)
    # create new column called LineId that contains line number starting from 1
    df['LineId'] = range(1, len(df)+1)
    # save to csv
    # check if directory exists
    if not os.path.exists(f'../Iudex/results_parsed_hyp_scan_wc/{dataset}'):
        os.makedirs(f'../Iudex/results_parsed_hyp_scan_wc/{dataset}')
    df.to_csv(f'../Iudex/results_parsed_hyp_scan_wc/{dataset}/{predictedPath}_parsed.csv', index=False)
    print('Saved to csv')

def extract_log_templates(text):
    return re.findall(r'LogTemplate\[\d+]: `([^`]*)`', text)

def extract_log_templates_v2(text):
    # Define the regex pattern to match both cases
    pattern = r'LogTemplate\[\d+\]: `([^`]*)`|LogTemplate\[\d+\]: (.*)'

    # Use re.findall to capture matches
    matches = re.findall(pattern, text)

    # Extract the relevant part from the matches
    results = []
    for match in matches:
        # Choose the group that is not empty
        results.append(match[0] if match[0] else match[1])

    return results
def most_frequent_template(templates):
    if templates:
        template_counter = Counter(templates)
        most_common_template, _ = template_counter.most_common(1)[0]
        return re.sub(r'{.*?}', '<*>', most_common_template)
    return ""


def process_group(pattern_id_group, models):
    pattern_id, group = pattern_id_group
    res = get_chat_completion(group, models)
    out = {}
    for model, response in zip(models, res):
        print('model: ', model)
        print('response: ', response)
        log_templates = extract_log_templates_v2(response)
        print('log_templates: ', log_templates)
        out[model] = log_templates
    return pattern_id, out

def parse_templates_multiple(dataset, predictedPath, models, max_logs=10, num_threads=None):
    predictedPathComplete = f'../Iudex/results_new_hyp_scan_wc/{dataset}/{predictedPath}.csv'
    # Read CSV
    df = pd.read_csv(predictedPathComplete)
    print(f'Number of patterns: {df["PatternID"].nunique()}')

    # # sort content of each group by alphabetical order
    grouped_df = df.groupby('PatternID')['Content'].apply(lambda x: '\n'.join([f'Log[{i+1}]: `{log}`' for i, log in enumerate(x.head(max_logs))]))

    def evenly_spread_logs(logs, n):
        step = len(logs) / n
        return [logs[int(i * step)] for i in range(n)]

    # Apply the function to the grouped DataFrame
    # grouped_df = df.groupby('PatternID')['Content'].apply(lambda x: '\n'.join(
        # [f'Log[{i + 1}]: `{log}`' for i, log in enumerate(evenly_spread_logs(sorted(x), max_logs))]))

        # print the first value of the grouped_df
    print(grouped_df)

    # Automatically determine the number of threads to use
    if num_threads is None:
        num_threads = min(32, (os.cpu_count() or 1) + 4)
    # num_threads = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda x: process_group(x, models), grouped_df.items()))
    # Initialize a dictionary for each model
    model_results = {model: {} for model in models}
    for pattern_id, out in results:
        for model, templates in out.items():
            freq = most_frequent_template(templates)
            print(f'{model} template: ', freq)
            freq = correct_single_template(freq, dataset)
            print(f'{model} corrected template: ', freq)
            model_results[model][pattern_id] = freq

    # Create and save a DataFrame for each model
    for model in models:
        model_df = df.copy()
        model_df['EventTemplate'] = model_df['PatternID'].map(model_results[model])
        model_df['LineId'] = range(1, len(model_df) + 1)

        output_dir = f'../Iudex/results_parsed_hyp_scan_wc/{dataset}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = f'{output_dir}/{predictedPath}_parsed_{model}.csv'
        model_df.to_csv(output_path, index=False)
        print(f'Saved {model} results to CSV')



def process_group_v1(pattern_id_group):
    pattern_id, group = pattern_id_group
    res = get_chat_completion(group)
    res = re.sub(r'{.*?}', '<*>', res)
    print('template: ', res)
    return pattern_id, res

def parse_templates_concurrent_v1(dataset, predictedPath, model, max_logs=5, num_threads=None):
    predictedPathComplete = f'../Iudex/results_new_hyp_scan_wc/{dataset}/{predictedPath}.csv'
    # Read CSV
    df = pd.read_csv(predictedPathComplete)
    print(f'Number of patterns: {df["PatternID"].nunique()}')

    grouped_df = df.groupby('PatternID')['Content'].apply(lambda x: '\n'.join(x.head(max_logs)))
    print(grouped_df)

    templates_dict = {}

    if num_threads is None:
        # Automatically determine the number of threads to use
        num_threads = min(32, (os.cpu_count() or 1) + 4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_group_v1, grouped_df.items()))

    for pattern_id, template in results:
        templates_dict[pattern_id] = template

    df['EventTemplate'] = df['PatternID'].map(templates_dict)
    print(df)

    df['LineId'] = range(1, len(df) + 1)

    output_dir = f'../Iudex/results_parsed_hyp_scan_wc/{dataset}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = f'{output_dir}/{predictedPath}_parsed_{model}.csv'
    df.to_csv(output_path, index=False)
    print('Saved to CSV')


def meta_reflection(dataset, predictedPath, models):
    """function that takes in a dataframe with the predicted EventTemplates,
    then prompts the model to see if any of the templates should be combined"""
    output_dir = f'../Iudex/results_parsed_hyp_scan_wc/{dataset}'
    for model in models:
        templatePath = f'{output_dir}/{predictedPath}_parsed_{model}.csv'
        df = pd.read_csv(templatePath)
        print(df)
        # get all the unique EventTemplates
        unique_templates = df['EventTemplate'].unique()
        # combine into 1 string, where each template starts with LogTemplate[i]
        prompt = "".join([f"LogTemplate[{i}]: `{template}`\n" for i, template in enumerate(unique_templates, 1)])
        print('sent to model: ', prompt)
        response = get_chat_completion(prompt, [model], meta_analysis=True)
        response = response[0]
        print(response)
        log_templates = extract_log_templates_v2(response)
        # throw exception if length of log_templates is not equal to length of unique_templates
        if len(log_templates) != len(unique_templates):
            raise ValueError("Length of log_templates does not match length of unique_templates")
        # Map back to the original dataframe
        template_dict = {template: log_template for template, log_template in zip(unique_templates, log_templates)}
        df['EventTemplate'] = df['EventTemplate'].map(template_dict)
        output_dir = f'../Iudex/results_parsed_hyp_scan_wc/{dataset}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = f'{output_dir}/{predictedPath}_parsed_{model}.csv'
        df.to_csv(output_path, index=False)
        print(f'Saved {model} updated results to CSV')


def post_process_templates(dataset, path, models):
    for model in models:
        out_path = f'results_parsed_hyp_scan_wc/{dataset}/{path}_parsed_{model}.csv'
        df = pd.read_csv(out_path)
        # group by event template, then count the number of unique templates
        grouped_df = df.groupby('EventTemplate')['LineId'].count().reset_index()
        # apply postProcess to templates
        grouped_df['CorrectedEventTemplate'] = grouped_df['EventTemplate'].apply(correct_single_template, args=(dataset,))
        # save back to the original dataframe
        df = df.merge(grouped_df[['EventTemplate', 'CorrectedEventTemplate']], on='EventTemplate', how='left')
        df['EventTemplate'] = df['CorrectedEventTemplate']
        df.drop(columns=['CorrectedEventTemplate'], inplace=True)
        df.to_csv(out_path, index=False)

datasets = ['Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux', 'Mac',
            'OpenSSH', 'OpenStack', 'Proxifier', 'Spark', 'Thunderbird', 'Zookeeper']
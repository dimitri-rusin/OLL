# Format the output of the configurations with the format given by Nguyen
'''
{       "n":1000,
        "experiment": "dynamic_lbd",
        "max_evals": 10000, # cutoff time of each OneLL run during the tuning
        "tuning_budget": 5000,
        "tuning_time": 3600, # the true tuning time in seconds, but can be skipped if there's not information available
        "evaluation_results": [500,600,555,...], # result of each evaluation run of the best configuration found by irace
        "best_configuration" : {"fx": [0,1,2,3,...,999], "lbd": [1,2,3,...]} # lbd is 1 for 0<=f(x)<1, 2 for 1<=f(x)<2, etc
}
'''

from config import N, sizes, experiment_types, descent_rates, get_cutoff, experiment_multiples_dynamic_bin, get_bins, experiment_multiples_dynamic, experiment_multiples_static, N2, iterative_seeding_sizes, iterative_seeding_multiples, iterative_seeding_iterations, iterative_seeding_seeds, get_iter_bins, binning_with_dynamic_sizes, binning_with_dynamic_seeds, binning_with_dynamic_iterations, get_dynamic_theory_lbd, BinningWithDynamicStrategy
from decoder import IraceDecoder
import json
import os
import re
import logging

tuning_time = 0

data = []

def read_data_from_irace_output(file_name):
  decoder = IraceDecoder()
  with open(file_name) as f:
    for line in f:
      decoder.note_line(line)
  return decoder.end()

def read_evaluation_from_json(file_name):
  with open(file_name) as f:
    return json.load(f)

for experiment_type in experiment_types: 
  if experiment_type in ['dynamic_theory', 'dynamic', 'static', 'binning_comparison', 'binning_comparison_with_static', 'dynamic_with_static']:
    for i in range(N):
      n = sizes[i]
      if experiment_type == 'binning_comparison' or experiment_type == 'binning_comparison_with_static':
        for binning_rate in descent_rates:
          experiment = f'{experiment_type}{binning_rate}'
          max_evals = get_cutoff(n)
          tunning_budget = int(experiment_multiples_dynamic_bin[i] * n)
          bin, bin_lookup = get_bins(n, descent_rate=binning_rate)
          fx = [0]
          for a in bin:
            fx.append(fx[-1] + a) 
          fx = fx[:-1]
          invalid_data = False
          for path in os.listdir('irace_output'):
            file_name_match = f'irace_output_{experiment_type}_{n}_{experiment_multiples_dynamic_bin[i]}_{binning_rate}_.*\\.txt'
            if re.match(file_name_match, path):
              decoder = IraceDecoder()
              with open(os.path.join('irace_output', path)) as f:
                for line in f:
                  decoder.note_line(line)
              try:
                lbd = decoder.end()
                if invalid_data:
                  print("recovered from invalid data for " + path)
                  invalid_data = False
                if len(lbd) == 0:
                  print("invalid data for " + path)
                  invalid_data = True
                elif not invalid_data:
                  break
              except:
                print("unable to parse data for " + path)
                invalid_data = True

          for path in os.listdir('irace_output'):
            if re.match(f'performance_{experiment_type}_{n}_{experiment_multiples_dynamic_bin[i]}_{binning_rate}_.*\\.json', path):
              with open(os.path.join('irace_output', path)) as f:
                evaluation_result = json.load(f)
          if not invalid_data:
            try:
              if len(fx) != len(lbd):
                print(f"invalid data for {experiment_type} {n} {experiment_multiples_dynamic_bin[i]} {binning_rate}")
              else:
                data.append({
                  'n': n,
                  'experiment': experiment,
                  'max_evals': max_evals,
                  'tuning_budget': tunning_budget,
                  'tuning_time': 0,
                  'evaluation_results': evaluation_result,
                  'best_configuration': {'fx': fx, 'lbd': lbd}
                })
            except:
              logging.exception('')
      else:
        experiment = experiment_type
        tuning_time = 0
        if experiment_type == 'dynamic':
          experiment_multiple = experiment_multiples_dynamic[i] 
          fx = list(range(n))
        elif experiment_type == 'static': 
          experiment_multiple = experiment_multiples_static[i] 
          fx = [0]
        elif experiment_type == 'dynamic_theory':
          experiment_multiple = 0
          fx = [i for i in range(n)]
        elif experiment_type == 'dynamic_with_static':
          experiment_multiple = experiment_multiples_dynamic[i] 
          fx = list(range(n))
        else:
          raise NotImplementedError()
        tunning_budget = sizes[i] * experiment_multiple
        
        max_evals = get_cutoff(n)

        if experiment_type == 'dynamic_theory':
          max_evals = 0
        invalid_data = False
        if experiment_type == 'dynamic' or experiment_type == 'static' or experiment_type == 'dynamic_with_static':
          file_name_match = f'irace_output_{experiment_type}_{n}_{experiment_multiple}_.*\\.txt'
          file_not_found = True
          for path in os.listdir('irace_output'):
            if re.match(file_name_match, path):
              file_not_found = False
              decoder = IraceDecoder()
              with open(os.path.join('irace_output', path)) as f:
                for line in f:
                  decoder.note_line(line)
              try:
                lbd = decoder.end()
                if invalid_data:
                  print("recovered from invalid data for " + path)
                  invalid_data = False
                if len(lbd) == 0:
                  print("invalid data for " + path)
                  invalid_data = True
                elif not invalid_data:
                  break
              except:
                print("unable to parse data for " + path)
                invalid_data = True
          if file_not_found:
            print(f"File {experiment_type} {n} {experiment_multiple} not found, skipping")
            invalid_data = True
        elif experiment_type == 'dynamic_theory':
          lbd = [(n / (n-f_x))**(1/2) for f_x in range(n)]
        else:
          raise NotImplementedError()
        if experiment_type == 'dynamic' or experiment_type == 'static' or experiment_type == 'dynamic_theory':
          try: 
            with open(os.path.join('irace_output', f'performance_{n}_{experiment_multiples_dynamic[i]}_{experiment_multiples_static[i]}.json')) as f:
              r = json.load(f)
              if experiment_type == 'dynamic':
                evaluation_result = r[0] 
              elif experiment_type == 'static':
                evaluation_result = r[2] 
              elif experiment_type == 'dynamic_theory': 
                evaluation_result = r[4]
              else:
                raise NotImplementedError()
          except FileNotFoundError:
            print(f"file not found for {experiment_type} {n} {experiment_multiple}, skipping")
            invalid_data = True
        elif experiment_type == 'dynamic_with_static':
          for path in os.listdir('irace_output'):
            if re.match(f'performance_{experiment_type}_{n}_{experiment_multiples_dynamic[i]}_.*\\.json', path):
              with open(os.path.join('irace_output', path)) as f:
                evaluation_result = json.load(f)      
        else:
          raise NotImplementedError()
        if not invalid_data:
          try:
            if len(fx) != len(lbd):
              print(f"invalid data for {experiment_type} {n} {experiment_multiple}. SHOULD'T FAIL HERE. CHECK FOR POTENTIAL LOGIC BUG.")
            else:
              data.append({
                'n': n,
                'experiment': experiment,
                'max_evals': max_evals,
                'tuning_budget': tunning_budget,
                'tuning_time': 0,
                'evaluation_results': evaluation_result,
                'best_configuration': {'fx': fx, 'lbd': lbd}
              })
          except:
            print(f'exception for {experiment_type} {n} {experiment_multiple}') 
            logging.exception('')
  elif experiment_type in ['binning_with_defaults']:
    for i in range(N2):
      n = iterative_seeding_sizes[i]
      max_evals = get_cutoff(n)
      tuning_time = 0
      for j in range(iterative_seeding_iterations[i]):
        tunning_budget = iterative_seeding_sizes[i] * iterative_seeding_multiples[i][j]
        experiment = experiment_type + str(j)
        fx = get_iter_bins(n, j+1)[:-1]
        failed = False
        try:
          evaluation_result = read_evaluation_from_json(f"irace_output/performance_{experiment_type}_{n}_{iterative_seeding_multiples[i][j]}_{j}_{iterative_seeding_seeds[i][0][j]}_{iterative_seeding_seeds[i][1][j]}.json")
        except:
          print(f"no evalution data for {experiment_type} {n} {j}")
          failed = True
        try:
          lbd = read_data_from_irace_output(f"irace_output/irace_output_{experiment_type}_{n}_{iterative_seeding_multiples[i][j]}_{j}_{iterative_seeding_seeds[i][0][j]}.txt")
        except:
          print(f"no irace data for {experiment_type} {n} {j}")
          failed = True
        if not failed:
          data.append({
            'n': n,
            'experiment': experiment,
            'max_evals': max_evals,
            'tuning_budget': tunning_budget,
            'evaluation_results': evaluation_result,
            'best_configuration': {'fx': fx, 'lbd': lbd}
          })
  elif experiment_type in ['binning_with_dynamic_start', 'binning_with_dynamic_end', 'binning_with_dynamic_middle']:
    mm = {
      'binning_with_dynamic_start': BinningWithDynamicStrategy.start, 
      'binning_with_dynamic_end': BinningWithDynamicStrategy.end,
      'binning_with_dynamic_middle': BinningWithDynamicStrategy.middle
    }
    for i in range(N2):
      bin_count = binning_with_dynamic_iterations[i]
      n = binning_with_dynamic_sizes[i]
      experiment = experiment_type
      fx = get_iter_bins(n, bin_count)[:-1]
      lbd = get_dynamic_theory_lbd(n, bin_count, mm[experiment_type])
      try:
        evaluation_result = read_evaluation_from_json(f'irace_output/performance_{experiment_type}_{i}_{binning_with_dynamic_seeds[i]}.json')
      except:
        print(f"no evaluation result for {experiment_type} {i}")
      data.append({
        'n': n,
        'experiment': experiment,
        'max_evals': 0,
        'tuning_budget': 0,
        'tuning_time':0,
        'evaluation_results': evaluation_result,
        'best_configuration': {'fx': fx, 'lbd': lbd}
      })
      
  else: 
    raise NotImplementedError()

with open('main_data.json', 'w') as f:
  json.dump(data, f, indent=2)




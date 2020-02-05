import os
import baselines.her.experiment.config as config
from baselines.her.util import (save_weight, load_weight)
import pickle
import click
import json

@click.command()
@click.option('--env_name', type=str, default=None, help='env_name')
@click.option('--note', type=str, default=None, help='unique notes')
@click.option('--seed', type=str, default=None, help='seed')


def main(env_name, note, seed):
  params = config.DEFAULT_PARAMS
  if note:
        with open('params/'+env_name+'/'+note+'.json', 'r') as file:
            override_params = json.loads(file.read())
            params.update(**override_params)
  policy_file = params['load_weight']

  if policy_file:
    if type(policy_file) is list:
      i = 0
      print('\n')
      for item in policy_file:
        print(i, item)
        i += 1
      seed = int(seed)
      seed = seed + 1
      if seed:
        index = seed - 1
      else:
        index = int(input('\nWhich one from the list?\n'))
      policy_file = policy_file[index]
      print(index, policy_file)
    base = os.path.splitext(policy_file)[0]
    with open(policy_file, 'rb') as f:
        pretrain = pickle.load(f)
    pretrain_weights = save_weight(pretrain.sess)
    output_file = open(base+'_weight.pkl', 'wb')
    pickle.dump(pretrain_weights, output_file)
    output_file.close()

if __name__ == '__main__':
    main()
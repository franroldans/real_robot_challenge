import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    print('HER parser: {}'.format(parser))
    # the environment setting
    # parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='src/rrc_example_package/rrc_example_package/her/saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')

    args = parser.parse_args()
    print('HER args:\n{}'.format(args))
    return args

# define the arguments that will be used in the SAC
def get_sac_args():
    parse = argparse.ArgumentParser()
    #parse.add_argument('--env-name', type=str, default='HalfCheetah-v2', help='the environment name')
    parse.add_argument('--cuda', action='store_true', help='use GPU do the training')
    parse.add_argument('--seed', type=int, default=123, help='the random seed to reproduce results')
    parse.add_argument('--hidden-size', type=int, default=256, help='the size of the hidden layer')
    parse.add_argument('--train-loop-per-epoch', type=int, default=1, help='the training loop per epoch')
    parse.add_argument('--q-lr', type=float, default=3e-4, help='the learning rate')
    parse.add_argument('--p-lr', type=float, default=3e-4, help='the learning rate of the actor')
    parse.add_argument('--n-epochs', type=int, default=int(3e3), help='the number of total epochs')
    parse.add_argument('--epoch-length', type=int, default=int(1e3), help='the lenght of each epoch')
    parse.add_argument('--n-updates', type=int, default=int(1e3), help='the number of training updates execute')
    parse.add_argument('--init-exploration-steps', type=int, default=int(1e3), help='the steps of the initial exploration')
    parse.add_argument('--init-exploration-policy', type=str, default='gaussian', help='the inital exploration policy')
    parse.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the replay buffer')
    parse.add_argument('--batch-size', type=int, default=256, help='the batch size of samples for training')
    parse.add_argument('--reward-scale', type=float, default=1, help='the reward scale')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parse.add_argument('--log-std-max', type=float, default=2, help='the maximum log std value')
    parse.add_argument('--log-std-min', type=float, default=-20, help='the minimum log std value')
    parse.add_argument('--entropy-weights', type=float, default=0.2, help='the entropy weights')
    parse.add_argument('--tau', type=float, default=5e-3, help='the soft update coefficient')
    parse.add_argument('--target-update-interval', type=int, default=1, help='the interval to update target network')
    parse.add_argument('--update-cycles', type=int, default=int(1e3), help='how many updates apply in the update')
    parse.add_argument('--eval-episodes', type=int, default=10, help='the episodes that used for evaluation')
    parse.add_argument('--display-interval', type=int, default=1, help='the display interval')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the place to save models')
    parse.add_argument('--reg', type=float, default=1e-3, help='the reg term')
    parse.add_argument('--auto-ent-tuning', action='store_true', help='tune the entorpy automatically')
    parse.add_argument('--log-dir', type=str, default='logs', help='dir to save log information')
    parse.add_argument('--env-type', type=str, default=None, help='environment type')
    parse.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parse.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parse.add_argument('--clip-range', type=float, default=5, help='the clip range')


    args = parse.parse_args()
    print('HER args:\n{}'.format(args))
    return args

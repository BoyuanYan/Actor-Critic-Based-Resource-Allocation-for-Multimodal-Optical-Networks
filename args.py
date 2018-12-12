import argparse

parser = argparse.ArgumentParser(
    description='GRWA Training')

parser.add_argument('--mode', type=str, default='alg',
                    help='RWA执行的模式，alg表示使用ksp+FirstFit，learning表示CNN学习模式, fcl表示FC学习模式，lstml表示LSTM学习模式')
parser.add_argument('--algo', type=str, default='a2c',
                    help="a2c, ppo, acktr, default a2c")
parser.add_argument('--cnn', type=str, default='mobilenetv2',
                    help="用到的CNN网络，默认是mobilenetv2，还有simplenet，simplestnet, expandsimplenet, deepersimplenet,  alexnet, squeeze的选择")
parser.add_argument('--workers', type=int, default=16,
                    help='默认同步执行多少个游戏，默认值16')
parser.add_argument('--steps', type=float, default=10e6,
                    help="所有游戏进程的训练总共要进行的步骤数")
parser.add_argument('--save-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--save-interval', type=int, default=100,
                    help='save interval, one save per n updates (default: 100)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--cuda', type=str, default="False",
                    help="是否使用GPU进行运算。如果为True，表示在集群上进行运算，有分布式操作。")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model')
parser.add_argument('--reward', type=int, default=1,
                    help='业务到达的时候，执行分配成功的奖励')
parser.add_argument('--punish', type=int, default=-1,
                    help='业务到达的时候，执行分配失败的惩罚')
parser.add_argument('--dpi', type=int, default=72,
                    help="生成图像的dpi")
parser.add_argument('--line-width', type=float, default=1,
                    help="链路的粗细")
parser.add_argument('--node-size', type=float, default=0.1,
                    help="节点的大小")
parser.add_argument('--append-route', type=str, default="True",
                    help="是否将路由信息作为一个维度附加到图片中去")
parser.add_argument('--file-prefix', type=str, default="resources",
                    help="resources目录的相对位置，默认是resources")
parser.add_argument('--resume', type=str,
                    help="resume interrupted process")
# 模型对比相关参数
parser.add_argument('--comp', type=str, default="None",
                    help="在ModelCompare.py脚本中，进行模型对比的参数，默认是None，即不进行对比；选项还有states和random")

#  RWA相关参数
parser.add_argument('--net', type=str, default='6node.md',
                    help="网络拓扑图，默认在resources目录下搜索")
parser.add_argument('--wave-num', type=int, default=10,
                    help='拓扑中每条链路的波长数')
parser.add_argument('--rou', type=int, default=5,
                    help='业务到达的平均间隔，泊松分布')
parser.add_argument('--miu', type=int, default=100,
                    help='业务持续的平均时间，泊松分布')
parser.add_argument('--max-iter', type=int, default=1000,
                    help='一次episode中，分配的业务数量')
parser.add_argument('--k', type=int, default=1,
                    help='RWA算法中，采取ksp计算路由的k值')
parser.add_argument('--img-width', type=int, default=224,
                    help="生成的网络灰度图的宽度")
parser.add_argument('--img-height', type=int, default=224,
                    help="生成的网络灰度图的高度")
parser.add_argument('--weight', type=str, default='None',
                    help='计算路由的时候，以什么属性为权重')
parser.add_argument('--step-over', type=str, default='one_time',
                    help="步进的模式，one_time表示每调用一次step，执行一个时间步骤；one_service表示每调用一次step，执行到下一个service到达的时候。")
# RL算法相关参数
parser.add_argument('--num-steps', type=int, default=5,
                    help='number of forward steps in A2C (default: 5)')
parser.add_argument('--base-lr', type=float, default=7e-4,
                    help='起始learning rate值')
parser.add_argument('--lr-adjust', type=str, default='constant',
                    help='learning rate的调整策略，包括constant，exp，linear')
parser.add_argument('--alpha', type=float, default=0.99,
                    help='RMSprop optimizer apha (default: 0.99)')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='max norm of gradients (default: 0.5)')

parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--use-gae', type=str, default="False",
                    help='https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/issues/49')
parser.add_argument('--expand-factor', type=int, default=2,
                    help="expandsimplenet的横向扩张系数，默认是2")
parser.add_argument('--ppo-epoch', type=int, default=4,
                    help='number of ppo epochs (default: 4)')
parser.add_argument('--num-mini-batch', type=int, default=32,
                    help='number of batches for ppo (default: 32)')
parser.add_argument('--clip-param', type=float, default=0.2,
                    help='ppo clip parameter (default: 0.2)')

args = parser.parse_args()


import sys
import re

if re.match(r"win",sys.platform)!=None:
    sys_type="windows"
elif re.match(r"linux",sys.platform)!=None:
    sys_type="linux"
else:
    sys_type="unknown"
    print("Libs: Unknown system type")

if sys_type in ["windows"]:
    dir_cifar10='A:/Software_Projects/Dataset/CIFAR10'
    dir_mnist='A:/Software_Projects/Dataset/MNIST'
elif sys_type in ["linux"]:
    dir_cifar10='/data4/wangweifan/DataSet/CIFAR10'
    dir_mnist='/data4/wangweifan/DataSet/MNIST'
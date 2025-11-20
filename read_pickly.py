import pickle
from env.task_env import TaskEnv


# 방법 2: 한 줄로 (test.py에서 사용하는 방식)
priority = pickle.load(open('RALTestSet_test/env_0.pkl', 'rb'))

original = pickle.load(open('RALTestSet_zeropriority/env_0.pkl', 'rb'))

import pdb;pdb.set_trace()
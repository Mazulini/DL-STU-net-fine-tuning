import sys
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry

if __name__ == '__main__':
    sys.exit(plan_and_preprocess_entry())
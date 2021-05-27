#!/bin/bash
python -m finetune_molcle train_GC --dataset_name bace --reward_method mc_l_shapley --input_model_file "./models_molCLE/molCLE_zinc_standard_agent_11.pth"

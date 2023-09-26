# -*- coding: utf-8 -*-
# file: train_apc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
#              your custom dataset_utils should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################


from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

save_path = 'state_dict'
apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.SLIDE_LCF_BERT
apc_config_english.num_epoch = 10
apc_config_english.evaluate_begin = 2
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0.5
apc_config_english.seed = {42, 56, 1}
apc_config_english.log_step = 5
apc_config_english.l2reg = 0.0001
apc_config_english.dynamic_truncate = True
apc_config_english.srd_alignment = True

Laptop14 = ABSADatasetList.Laptop14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Laptop14,
                          checkpoint_save_mode=1,
                          auto_device=True
                          )

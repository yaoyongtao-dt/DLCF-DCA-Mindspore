# -*- coding: utf-8 -*-
# file: change_loading_device.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa import load_sentiment_classifier, APCCheckpointManager
import x2ms_adapter

model_path = APCCheckpointManager.get_checkpoint('Chinese')
sent_classifier = load_sentiment_classifier(checkpoint=model_path,
                                            auto_device=True,  # Use CUDA if available
                                            )
# The default loading device is CPU， if auto_device=True will load model on CUDA if any
# you can load the model to CPU manually
sent_classifier

# load the model to CUDA (0)
sent_classifier

# # load the model to CPU or CUDA, like cpu, cuda:0, cuda:1, main.
x2ms_adapter.to(sent_classifier, 'cuda:0')

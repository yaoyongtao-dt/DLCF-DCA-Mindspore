from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

apc_config_english = APCConfigManager.get_apc_config_english()

# BERT_SPC DLCF_DCA_BERT
apc_config_english.model = APCModelList.DLCF_DCA_BERT
apc_config_english.lcf = "cdm" # or "cdw"
apc_config_english.dlcf_a = 2
apc_config_english.dca_p = 1
apc_config_english.dca_layer = 3
apc_config_english.dropout = 0.5
apc_config_english.num_epoch = 10
apc_config_english.l2reg = 0.00001
apc_config_english.seed = {0, 1, 2, 3}
apc_config_english.evaluate_begin = 0

dataset_path = ABSADatasetList.Restaurant14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=dataset_path,  # train set and test set will be automatically detected
                          checkpoint_save_mode=1,  # =None to avoid save model
                          auto_device=True  # automatic choose CUDA or CPU
                          )

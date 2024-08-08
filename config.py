from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r100"
config.resume = False
config.save_all_states = True
config.output = "./output"
config.embedding_size = 512
config.sample_rate = 0.2
config.interclass_filtering_threshold = 0

config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 32
config.gradient_acc = 12
config.optimizer = "sgd"

config.lr = 0.3
config.verbose = 200
config.num_workers  = 32

config.num_epoch = 200
config.warmup_epoch = 1

config.frequent = 20
config.warmup_step = 10

config.seed = 500

# dir of data train
config.rec = "/home/thainq97/dev/insightface/dataset/101_ObjectCategories"
# dir of data val

config.val_dir = "/home/thainq97/dev/insightface/dataset/101_ObjectCategories"

# wandb
config.using_wandb = True
config.wandb_entity = "sonnguyen222k"
config.wandb_key = "44b383067a5a3c615dbec0050b65c1950147c6ff"
config.wandb_project = "Training Face Recognition"
config.wandb_resume = False
config.notes = "test222"
config.suffix_run_name = "test3333"
config.wandb_log_all = True
config.save_artifacts = False




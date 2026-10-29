
# 训练数据路径设置
pretrained_model = "./sd-models/model-plaidshirtprogrammer.ckpt"  # 预训练模型路径（这里是使用dreambooth训练的格子衫模型）
is_v2_model = 0                             # SD2.0模型标志 (0表示SD1.0，1表示SD2.0)
parameterization = 0                        # 参数化标志，实验性功能
train_data_dir = "./train/lora-glass"        # 训练数据集路径
reg_data_dir = ""                            # 正则化图像目录，默认不使用正则化图像

# 网络设置
network_module = "networks.lora"             # 网络模块，LoRA训练为默认值"networks.lora"
network_weights = ""                         # LoRA网络预训练权重路径
network_dim = 32                             # 网络维度，通常为4到128之间
network_alpha = 32                           # 网络alpha参数，通常与network_dim相同或较小以防止下溢，默认为1


# 训练相关参数
resolution = "512,512"       # 图像分辨率，宽和高。支持非正方形，但必须是 64 的倍数。
batch_size = 1               # 每个训练批次的样本数量。
max_train_epoches = 10       # 最大训练轮数。
save_every_n_epochs = 2      # 每隔 N 个轮次保存一次模型。

# 训练模式：
train_unet_only = 0          # 是否仅训练 U-Net。开启此选项可能会减少显存使用，但可能会牺牲模型效果。
train_text_encoder_only = 0  # 是否仅训练文本编码器。
stop_text_encoder_training = 0 # 在第 N 步时停止训练文本编码器。

# 噪声和标记处理：
noise_offset = "0"           # 在训练中添加噪声偏移，以改善生成非常暗或非常亮的图像。如果启用，建议参数为 0.1。
keep_tokens = 0              # 在随机打乱标记时，保留前 N 个不变的标记。
min_snr_gamma = 0            # 伽马射线事件的最小信噪比（SNR）值，默认为 0。

# 学习率设置
lr = "1e-4"                     # 整体学习率，用于控制整个模型的学习速率
unet_lr = "1e-4"                 # U-Net 部分的学习率
text_encoder_lr = "1e-5"         # 文本编码器的学习率
lr_scheduler = "cosine_with_restarts"  # 学习率调度器的类型，选择了余弦退火重启模式，可选还包括 "linear", "cosine", "polynomial", "constant", "constant_with_warmup", "adafactor"
lr_warmup_steps = 0              # 学习率预热步数，当学习率调度器为 "constant" 或 "adafactor" 时，该值需要设为 0。
lr_restart_cycles = 1            # 余弦退火的重启周期数，仅在学习率调度器为 "cosine_with_restarts" 时生效。

# 优化器设置
optimizer_type = "AdamW8bit"     # 优化器类型，默认为 8bitadam，其他可选项包括 "AdamW", "AdamW8bit",


# 输出设置
output_name = "glass"            # 输出模型的名称，用于保存训练得到的模型。
save_model_as = "safetensors"  # 模型保存的格式，可选为 "ckpt"、"pt"、"safetensors" 等。

# 恢复训练设置
save_state = 0                 # 是否保存训练状态，如果设置为1，将在训练期间保存训练状态，例如模型的epoch数。
resume = ""                    # 从指定状态文件夹中恢复训练。需要与上述参数同时使用。因规范文件限制，epoch数和全局步数不会保存，即使在恢复时也会从1开始。

# 其他设置
min_bucket_reso = 256           # 任意最小分辨率，用于某些处理中的最小分辨率限制。
max_bucket_reso = 1024          # 任意最大分辨率，用于某些处理中的最大分辨率限制。
persistent_data_loader_workers = 0  # 持久化数据加载器的工作线程数，可以减少每个epoch之间的停顿，但需要注意可能导致内存占用较大。
clip_skip = 2                   # clip skip 参数，用于梯度裁剪。

# 启动加速器，进行模型训练
accelerate launch ${launchArgs[@]} --num_cpu_threads_per_process=8 "./sd-scripts/train_network.py" \
  --enable_bucket \
  --pretrained_model_name_or_path=$pretrained_model \
  --train_data_dir=$train_data_dir \
  --output_dir="./output" \
  --logging_dir="./logs" \
  --log_prefix=$output_name \
  --resolution=$resolution \
  --network_module=$network_module \
  --max_train_epochs=$max_train_epoches \
  --learning_rate=$lr \
  --unet_lr=$unet_lr \
  --text_encoder_lr=$text_encoder_lr \
  --lr_scheduler=$lr_scheduler \
  --lr_warmup_steps=$lr_warmup_steps \
  --lr_scheduler_num_cycles=$lr_restart_cycles \
  --network_dim=$network_dim \
  --network_alpha=$network_alpha \
  --output_name=$output_name \
  --train_batch_size=$batch_size \
  --save_every_n_epochs=$save_every_n_epochs \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --seed="1337" \
  --cache_latents \
  --prior_loss_weight=1 \
  --max_token_length=225 \
  --caption_extension=".txt" \
  --save_model_as=$save_model_as \
  --min_bucket_reso=$min_bucket_reso \
  --max_bucket_reso=$max_bucket_reso \
  --keep_tokens=$keep_tokens \
  --xformers --shuffle_caption ${extArgs[@]}

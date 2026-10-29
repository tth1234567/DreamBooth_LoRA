# 创建目录，确保 Hugging Face Accelerated Inference API 的配置文件目录存在
mkdir -p ~/.cache/huggingface/accelerate

# 将默认的配置信息写入配置文件 default_config.yaml
cat > ~/.cache/huggingface/accelerate/default_config.yaml <<- EOM
compute_environment: LOCAL_MACHINE  # 计算环境设为本地机器
deepspeed_config: {}  # DeepSpeed 配置为空
distributed_type: 'NO'  # 不使用分布式训练
downcast_bf16: 'no'  # 不使用 bf16 数据类型
fsdp_config: {}  # FSDP 配置为空
machine_rank: 0  # 机器的排名设为 0
main_process_ip: null  # 主进程 IP 设为空
main_process_port: null  # 主进程端口设为空
main_training_function: main  # 主训练函数设为 main
mixed_precision: fp16  # 使用混合精度 fp16
num_machines: 1  # 使用的机器数设为 1
num_processes: 1  # 使用的进程数设为 1
use_cpu: false  # 不使用 CPU
EOM
